"""Follow-up changes to the program ASTA just built, and approvals that were
answered before they were read out."""

from __future__ import annotations

import queue
import threading
import time

from core.contracts import IntentType, TaskStatus
from core.planner import Planner
from speech import speech_module
from tests.test_desktop_program import (
    GAME, GAME_V2, GOAL, FakePopen, _approve_all, _empty_project, _fix_plan, _game_plan, _with_program_tool,
)
from tests.test_project_edit_fix_loop import _runtime, _shutdown

CANVAS_KEYS = (
    "import tkinter as tk\n"
    "class Game:\n"
    "    def __init__(self):\n"
    "        self.canvas = tk.Canvas(width=400, height=400)\n"
    "        self.canvas.pack()\n"
    "        self.canvas.bind('<Left>', lambda e: None)\n"
    "game = Game()\n"
    "game.canvas.mainloop()\n"
)


def _open(content, path="snake.py"):
    return [{"tool": "filesystem.write_file", "path": path, "content": content},
            {"tool": "project.open_program", "path": path}]


# --- desktop programs are checked before they are written ----------------------

def test_keys_bound_on_a_canvas_are_caught_before_opening():
    error = Planner._desktop_program_error(_open(CANVAS_KEYS))
    assert error.startswith("Desktop programs must receive their keys")
    assert "self.canvas" in error
    # Bound on the root, with bind_all or with focus_set: fine.
    assert Planner._desktop_program_error(_open(GAME + "root.bind('<w>', print)\n")) == ""
    assert Planner._desktop_program_error(_open(CANVAS_KEYS.replace("game = Game()", "game = Game()\ngame.canvas.focus_set()"))) == ""
    assert Planner._desktop_program_error(_open(
        "import tkinter as tk\nclass App(tk.Tk):\n    def __init__(self):\n        super().__init__()\n"
        "        self.bind('<Up>', print)\n")) == ""
    # Checked once: a corrected plan that still does it goes through.
    assert Planner._desktop_program_error(_open(CANVAS_KEYS), strict=False) == ""
    # Scripts that are only run in the sandbox aren't checked.
    assert Planner._desktop_program_error(_open(CANVAS_KEYS)[:1]) == ""


def test_a_desktop_program_must_be_valid_python():
    error = Planner._desktop_program_error(_open("def f():\n  x = 1\n    return x\n"), strict=False)
    assert error.startswith("Desktop programs must be valid Python: snake.py line 3")


def test_a_game_whose_keys_cannot_work_is_replanned_once(tmp_path):
    workspace = _empty_project(tmp_path)
    run = _runtime(tmp_path, workspace, _game_plan(CANVAS_KEYS), _game_plan(GAME))
    popen = FakePopen((None, ""))
    _with_program_tool(run, popen)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        task = _approve_all(run)
        assert task.status is TaskStatus.COMPLETED, task.error
    finally:
        _shutdown(run)
    assert "receive their keys" in run.brain.intents[1]["planning_feedback"]
    assert (workspace / "snake.py").read_text() == GAME


# --- "make the snake move with W, A, S, D also" ---------------------------------

def _wait_for_second_task(run, answered, deadline=10.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        for request in list(run.confirmations):
            if request.request_id not in answered:
                answered.add(request.request_id)
                run.kernel.event_bus.emit("tool_confirmation_response", request_id=request.request_id, approved=True)
        tasks = sorted(run.kernel.task_manager.list(), key=lambda t: t.created_at)
        if len(tasks) > 1 and tasks[-1].status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return tasks[-1]
        time.sleep(0.02)
    raise AssertionError("the follow-up task never finished")


def test_a_follow_up_change_edits_the_game_and_reopens_it(tmp_path):
    workspace = _empty_project(tmp_path)
    run = _runtime(tmp_path, workspace, _game_plan(GAME), _fix_plan(GAME_V2))
    popen = FakePopen((None, ""))
    _with_program_tool(run, popen)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        assert _approve_all(run).status is TaskStatus.COMPLETED
        answered = {request.request_id for request in run.confirmations}
        run.kernel.event_bus.emit("user_message", "Okay make the snake move with W, A, S, D also.")
        task = _wait_for_second_task(run, answered)
        assert task.status is TaskStatus.COMPLETED, task.error
    finally:
        _shutdown(run)
    goal = run.brain.goals[1]
    assert goal.startswith("In snake.py: make the snake move with W, A, S, D also.")
    assert "open snake.py on the desktop again" in goal
    assert (workspace / "snake.py").read_text() == GAME_V2
    assert len(popen.calls) == 2


def _task(goal, tools, *, intent="task", age=5.0):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    steps = [SimpleNamespace(metadata={"tool": tool, "path": "snake.py"}) for tool in tools]
    return SimpleNamespace(
        id="t1", goal=goal, metadata={"intent_type": intent}, plan=SimpleNamespace(steps=steps),
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=age),
    )


class _Tasks:
    def __init__(self, *tasks):
        self.tasks = tasks

    def list(self):
        return list(self.tasks)


def test_follow_up_routing_needs_a_recent_project_task_and_a_change_to_it():
    from core.agent.project_followup import followup_intent

    game = _Tasks(_task(GOAL, ["filesystem.write_file", "filesystem.read_file", "project.open_program"]))
    routed = followup_intent("Make the snake faster", game)
    assert routed.intent is IntentType.TASK and routed.classifier == "project_followup"
    assert "project.open_program" in routed.entities["capability_scope"]
    assert followup_intent("can you add a score to it", game) is not None
    assert followup_intent("change snake.py so walls wrap around", game) is not None
    # Not a change, or not about that work.
    assert followup_intent("what is the capital of France", game) is None
    assert followup_intent("make me a sandwich", game) is None
    assert followup_intent("play Baithi Hai on Spotify", game) is None
    # Only right after a project task.
    assert followup_intent("make it faster", _Tasks(_task(GOAL, ["project.open_program"], age=4000))) is None
    assert followup_intent("make it louder", _Tasks(_task("play music", ["media.control"], intent="command"))) is None
    assert followup_intent("make it faster", _Tasks()) is None


# --- approvals answered before they were read out ---------------------------------

def _speech():
    module = object.__new__(speech_module.SpeechModule)
    module._state_lock = threading.Lock()
    module._queue = queue.Queue()
    module._defer_speech = False
    module._deferred_sentences = []
    module._queued_text = 0
    module._synthesis_inflight = 0
    module._speech_active = False
    module._presentation_mode_active = False
    return module


def test_an_answered_question_is_not_read_out():
    module = _speech()
    question = "It's written. Can I open snake.py on your desktop? It runs outside the sandbox."
    for line in (question, "Done. It's up on your desktop, give it a try."):
        module._queue.put(line)
        module._queued_text += 1
    assert module.on_speech_retract(text=question) == 1
    assert list(module._queue.queue) == ["Done. It's up on your desktop, give it a try."]
    assert module._queued_text == 1
    assert module.on_speech_retract(text=question) == 0


def test_answering_an_approval_retracts_its_spoken_question(tmp_path):
    workspace = _empty_project(tmp_path)
    run = _runtime(tmp_path, workspace, _game_plan(GAME))
    _with_program_tool(run, FakePopen((None, "")))
    retracted = []
    run.kernel.event_bus.subscribe("speech_retract", lambda text=None, **kw: retracted.append(text))
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        assert _approve_all(run).status is TaskStatus.COMPLETED
    finally:
        _shutdown(run)
    assert len(retracted) == 1 and "snake.py" in retracted[0]
