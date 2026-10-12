"""ASTA says the few lines that matter, naturally; the chat shows everything."""

from __future__ import annotations

import random
import threading
import time
from types import SimpleNamespace

import numpy as np

from core.autonomy.sandbox_setup import SandboxSetupResult, SandboxSetupStatus
from core.contracts import TaskStatus
from core.phrasing import LINES, Phrasebook, acknowledgement
from core.task_runtime import TaskRuntimeModule
from core.tools.workspace import edit_requested, planning_file_context
from speech import speech_module
from tests.test_project_edit_fix_loop import FIXED, GOAL
from tests.test_sandbox_self_repair import (
    StoppedThenStartedSandbox,
    _project,
    _runtime,
    _wait_terminal,
)
from voice.echo_gate import EchoGate, voiced_fraction


def _pool(kind, **slots):
    return {line.format(**slots) for line in LINES[kind]}


# --- what is said ------------------------------------------------------------

def _deferring_speech():
    module = object.__new__(speech_module.SpeechModule)
    module._state_lock = threading.Lock()
    module._defer_speech = True  # queued lines land in _deferred_sentences
    module._deferred_sentences = []
    return module


def test_speech_says_the_short_line_and_skips_show_only_text():
    speech = _deferring_speech()
    speech.on_assistant_sentence("Done. Updated `calc.py` (+1 -1 lines) ...", spoken="All done. Check it out.")
    speech.on_assistant_sentence("The code sandbox is ready. Continuing your task.", spoken="")
    speech.on_assistant_sentence("Plain reply.")
    assert speech._deferred_sentences == ["All done. Check it out.", "Plain reply."]


def test_phrases_vary_and_never_repeat_back_to_back():
    book = Phrasebook(rng=random.Random(7))
    picks = [book.say("done_tests") for _ in range(12)]
    assert len(set(picks)) >= 4
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert set(picks) <= _pool("done_tests")


def test_acknowledgement_sounds_like_a_person():
    fix = acknowledgement("Fix the bug in calc.py")
    assert fix in _pool("ack_fix") | _pool("ack")
    assert acknowledgement("Create a hello world script") in _pool("ack")


def test_closing_line_says_the_work_is_done():
    report = TaskRuntimeModule._spoken_report
    changed = [{"path": "calc.py", "lines_added": 1, "lines_removed": 1}]
    assert report(changed, {"stdout": "ALL TESTS PASSED"}, command_step_arguments=["python", "test_calc.py"],
                  attempts=1, prefix_notes=[]) in _pool("done_tests")
    assert report([], {"stdout": "ALL TESTS PASSED"}, command_step_arguments=["python", "test_calc.py"],
                  attempts=1, prefix_notes=[]) in _pool("done_already_passing")
    line = report(changed, {"stdout": "hello"}, command_step_arguments=["python", "main.py"],
                  attempts=2, prefix_notes=[])
    assert any(line.startswith(done) for done in _pool("done"))
    assert "hello" in line


def test_spoken_approval_names_python_not_the_interpreter_path():
    from ai.ai_module import AIModule

    for target in ("python.exe", r"E:\\Projects\\ASTA\\.venv\\Scripts\\python.exe", "/usr/bin/python3.13", "python3"):
        request = SimpleNamespace(metadata={}, tool="system.run_command",
                                  arguments={"target": target, "arguments": ["test_calc.py"]})
        assert AIModule._spoken_approval(request, False) in _pool("approve_run", command="python test_calc.py"), target


def test_a_repaired_fix_task_speaks_only_the_main_lines(tmp_path):
    workspace = _project(tmp_path)
    sandbox = StoppedThenStartedSandbox()

    def start_docker(context):
        sandbox.engine_running = True
        return SandboxSetupResult(
            SandboxSetupStatus.COMPLETED,
            "The code sandbox is ready (started Docker Desktop). Continuing your task.",
            {"observed": ["docker engine is not running"], "actions": ["started Docker Desktop"]},
        )

    run = _runtime(tmp_path, workspace, sandbox, start_docker)
    shown, heard = [], []

    def record(text=None, spoken=None, **kw):
        shown.append(text)
        if spoken is None:
            heard.append(text)
        elif spoken.strip():
            heard.append(spoken)

    run.kernel.event_bus.subscribe("assistant_sentence", record)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        run.kernel.event_bus.emit(
            "tool_confirmation_response", request_id=run.confirmations[0].request_id, approved=True,
        )
        task = _wait_terminal(run.kernel)
        assert task.status is TaskStatus.COMPLETED, task.error
        assert (workspace / "calc.py").read_text(encoding="utf-8") == FIXED
    finally:
        for module in run.modules:
            module.shutdown()

    assert any("sandbox is ready" in line for line in shown)  # shown...
    assert not any("sandbox is ready" in line for line in heard)  # ...not said
    assert any(line in _pool("approve_run", command="python test_calc.py") for line in heard), heard
    assert heard[-1] in _pool("done_tests"), heard
    assert all(len(line) <= 120 for line in heard), heard


# --- not being cut off by its own voice ----------------------------------------

def _speech_like_echo(seconds, level, rng):
    """Frame energies of synthetic speech: syllables, pauses, loud words."""
    frames = []
    for index in range(int(seconds / 0.04)):
        syllable = 0.25 + 0.75 * abs(np.sin(index * 0.9)) * rng.uniform(0.6, 1.0)
        if index % 17 in (0, 1):
            syllable *= 0.1  # a pause between words
        if index % 29 in (10, 11, 12, 13):
            syllable *= 1.6  # an emphasised word
        frames.append((level * syllable, level * syllable * 3.5))
    return frames


def test_own_voice_with_pauses_and_emphasis_does_not_interrupt():
    rng = random.Random(3)
    for level in (0.01, 0.03, 0.08):
        gate = EchoGate(tts_confirm_frames=6)
        frames = _speech_like_echo(8.0, level, rng)
        warm = int(0.45 / 0.04)
        hits = [gate.update(r, p, tts_active=True, warming=i < warm) for i, (r, p) in enumerate(frames)]
        assert not any(hits), level


def test_user_talking_over_asta_still_interrupts():
    rng = random.Random(4)
    gate = EchoGate(tts_confirm_frames=6)
    frames = _speech_like_echo(2.0, 0.02, rng)
    for i, (r, p) in enumerate(frames):
        assert not gate.update(r, p, tts_active=True, warming=i < 11)
    user = [gate.update(0.12, 0.45, tts_active=True) for _ in range(8)]
    assert any(user)
    assert user.index(True) >= 5  # needs ~240 ms, not 80 ms


def test_idle_barge_in_stays_quick():
    gate = EchoGate()
    assert not gate.update(0.05, 0.2, tts_active=False)
    assert gate.update(0.05, 0.2, tts_active=False)


def test_voiced_fraction_tells_silence_from_talking():
    rate = 16000
    silence = np.random.default_rng(1).normal(0, 0.002, rate)
    talk = np.sin(np.arange(rate) / 8.0) * 0.1
    assert voiced_fraction(silence, rate) < 0.15
    assert voiced_fraction(talk, rate) > 0.8
    assert voiced_fraction(np.zeros(100), rate) is None


def _speech_with_queue():
    module = object.__new__(speech_module.SpeechModule)
    module._state_lock = threading.Lock()
    module._defer_speech = False
    module._deferred_sentences = []
    module._interrupt_event = threading.Event()
    module._queued_text = 0
    module._speech_active = True
    module._presentation_mode_active = False
    module._current_text = "Everything's in place. Can I run python test_calc.py now?"
    import queue

    module._queue = queue.Queue()
    module.event_bus = SimpleNamespace(emit=lambda *a, **k: None)
    module._publish_audio_level = lambda *a, **k: None
    said = []
    module.on_assistant_sentence = lambda text, spoken=None, _replay=False: said.append((text, _replay))
    return module, said


def test_a_sentence_cut_by_asta_own_echo_is_said_again_once():
    speech, said = _speech_with_queue()
    speech.on_speech_interrupt()
    speech.on_speech_false_interrupt()
    assert said == [("Everything's in place. Can I run python test_calc.py now?", True)]
    speech.on_speech_false_interrupt()  # nothing left to repeat
    assert len(said) == 1


def test_no_repeat_when_something_newer_was_said_or_too_late():
    speech, said = _speech_with_queue()
    speech.on_speech_interrupt()
    speech._sentences_since_cut = 1  # a newer line arrived after the cut
    speech.on_speech_false_interrupt()
    speech, said2 = _speech_with_queue()
    speech.on_speech_interrupt()
    speech._cut_at = time.monotonic() - 60
    speech.on_speech_false_interrupt()
    assert said == [] and said2 == []


# --- a test named as the finish line is not permission to rewrite it ---------

def test_test_file_named_as_the_goal_is_read_only(tmp_path):
    goal = "Fix the bug in calc.py so the tests in test_calc.py pass"
    assert edit_requested(goal, "calc.py")
    assert not edit_requested(goal, "test_calc.py")
    assert edit_requested("The assertion is wrong, fix test_calc.py", "test_calc.py")
    assert edit_requested("Add a subtraction test to test_calc.py", "test_calc.py")
    assert not edit_requested("Make test_calc.py pass", "test_calc.py")

    from core import Kernel

    workspace = _project(tmp_path)
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    context = planning_file_context(kernel.workspace_manager, goal)
    flags = {entry["path"]: entry["read_only"] for entry in context["file_contents"]}
    assert flags == {"calc.py": False, "test_calc.py": True}
