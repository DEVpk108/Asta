"""ASTA says only the few lines that matter; the chat shows everything."""

from __future__ import annotations

import threading

from core.contracts import TaskStatus
from core.task_runtime import TaskRuntimeModule
from speech import speech_module
from tests.test_project_edit_fix_loop import FIXED, GOAL
from tests.test_sandbox_self_repair import (
    StoppedThenStartedSandbox,
    _project,
    _runtime,
    _wait_terminal,
)
from core.autonomy.sandbox_setup import SandboxSetupResult, SandboxSetupStatus


def _deferring_speech():
    module = object.__new__(speech_module.SpeechModule)
    module._state_lock = threading.Lock()
    module._defer_speech = True  # queued lines land in _deferred_sentences
    module._deferred_sentences = []
    return module


def test_speech_says_the_short_line_and_skips_show_only_text():
    speech = _deferring_speech()
    speech.on_assistant_sentence("Done. Updated `calc.py` (+1 -1 lines) ...", spoken="Done. I updated calc.py.")
    speech.on_assistant_sentence("The code sandbox is ready. Continuing your task.", spoken="")
    speech.on_assistant_sentence("Plain reply.")
    assert speech._deferred_sentences == ["Done. I updated calc.py.", "Plain reply."]


def test_spoken_report_is_short():
    report = TaskRuntimeModule._spoken_report
    changes = [{"path": "calc.py", "lines_added": 1, "lines_removed": 1}]
    assert report(changes, {"stdout": "ALL TESTS PASSED"}, command_step_arguments=["python", "test_calc.py"],
                  attempts=1, prefix_notes=[]) == "Done. I updated calc.py, and the tests passed."
    assert report([], {"stdout": "hello"}, command_step_arguments=["python", "main.py"],
                  attempts=2, prefix_notes=[]) == "Done. It printed hello. It took 2 tries."


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
    assert any(line.startswith("Can I run python test_calc.py?") for line in heard), heard
    assert heard[-1] == "Done. I updated calc.py, and the tests passed.", heard
    assert all(len(line) <= 90 for line in heard), heard


def test_spoken_approval_says_python_not_the_interpreter_path():
    from types import SimpleNamespace
    from ai.ai_module import AIModule

    for target in ("python.exe", r"E:\\Projects\\ASTA\\.venv\\Scripts\\python.exe", "/usr/bin/python3.13", "python3"):
        request = SimpleNamespace(metadata={}, tool="system.run_command",
                                  arguments={"target": target, "arguments": ["test_calc.py"]})
        assert AIModule._spoken_approval(request, False) == "Can I run python test_calc.py?", target
