"""Project work asked for in conversation: the model reads the sentence with
the project, its files and the last turns, and starts a task or asks."""

from __future__ import annotations

import json

from core.agent.project_conversation import ProjectConversation, ground_goal, parse_decision
from core.contracts import TaskStatus
from tests.test_desktop_program import GAME, GAME_V2, FakePopen, _fix_plan, _with_program_tool
from tests.test_project_edit_fix_loop import _runtime, _shutdown
import time


class ScriptedModel:
    """complete_json that answers from a list and records what it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def generate_response(self, *args, **kwargs):
        raise AssertionError("the chat model is not needed here")

    def complete_json(self, system, user, max_tokens=96, timeout=None):
        self.prompts.append(user)
        return json.dumps(self.answers.pop(0))


def _snake_project(tmp_path):
    workspace = tmp_path / "asta-demo-snake"
    workspace.mkdir()
    (workspace / "snake.py").write_text(GAME)
    return workspace


def test_a_vague_request_gets_a_question_and_the_answer_starts_the_task(tmp_path):
    workspace = _snake_project(tmp_path)
    run = _runtime(tmp_path, workspace, _fix_plan(GAME_V2))
    popen = FakePopen((None, ""))
    _with_program_tool(run, popen)
    ai = run.modules[1]
    model = ai.engine = ScriptedModel(
        {"action": "ask", "goal": "", "question": "Sure. What would you like to change in snake.py?"},
        {"action": "task", "goal": "In snake.py, make the snake also move with W, A, S and D. "
                                   "Then open snake.py on the desktop.", "question": ""},
    )
    chat = []
    ai.chat_history.messages_for_session = lambda session_id, limit=12: list(chat)
    try:
        # A fresh start: no task has run yet, only the project on disk.
        said = "Hey I want you to make changes in the snake game we were building"
        chat.append({"role": "user", "text": said})
        run.kernel.event_bus.emit("user_message", said)
        assert run.spoken[-1] == "Sure. What would you like to change in snake.py?"
        assert run.kernel.task_manager.list() == []
        chat.append({"role": "assistant", "text": run.spoken[-1]})

        run.kernel.event_bus.emit("user_message", "Let it move with W A S D too")
        task = None
        end = time.monotonic() + 10
        answered = set()
        while time.monotonic() < end:
            for request in list(run.confirmations):
                if request.request_id not in answered:
                    answered.add(request.request_id)
                    run.kernel.event_bus.emit("tool_confirmation_response", request_id=request.request_id, approved=True)
            tasks = run.kernel.task_manager.list()
            if tasks and tasks[0].status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                task = tasks[0]
                break
            time.sleep(0.02)
        assert task is not None and task.status is TaskStatus.COMPLETED, task and task.error
    finally:
        _shutdown(run)
    first, second = model.prompts
    assert "Project: calc" in first and "Project files: snake.py" in first
    assert "User said: Hey I want you to make changes" in first
    assert "A.S.T.A. just asked: Sure. What would you like to change in snake.py?" in second
    assert "A.S.T.A.: Sure. What would you like" in second  # the turn itself, from chat history
    assert run.brain.goals[0].startswith("In snake.py, make the snake also move with W, A, S and D.")
    assert (workspace / "snake.py").read_text() == GAME_V2
    assert len(popen.calls) == 1


def test_small_talk_never_pays_for_the_project_check(tmp_path):
    workspace = _snake_project(tmp_path)
    talk = ProjectConversation()
    files = ["snake.py"]
    assert talk.worth_checking("I want you to make changes in the snake game", files)
    assert talk.worth_checking("can you add a pause key to snake.py", files)
    assert talk.worth_checking("could you build a calculator app", files)
    assert not talk.worth_checking("how are you today", files)
    assert not talk.worth_checking("what's the weather like", files)
    assert not talk.worth_checking("yes", files)
    # After a question, the next answer is always read with it.
    talk.pending = {"question": "What should change?", "until": talk.clock() + 60}
    assert talk.worth_checking("yeah add that", files)


def test_decisions_are_parsed_strictly():
    assert parse_decision('{"action": "task", "goal": "In snake.py add WASD"}').goal == "In snake.py add WASD"
    assert parse_decision('{"action": "ask", "question": "What should change?"}').action == "ask"
    assert parse_decision('{"action": "task", "goal": ""}').action == "none"  # no goal, no task
    assert parse_decision('{"action": "delete_everything"}') is None
    assert parse_decision("not json") is None


def test_the_file_the_user_meant_is_named_in_the_goal():
    assert ground_goal("Make the snake faster", "make the snake faster", ["snake.py", "README.md"]) == (
        "Make the snake faster (in snake.py).")
    assert ground_goal("In snake.py, add a score", "add a score", ["snake.py"]) == "In snake.py, add a score"
    assert ground_goal("Add a score", "add a score", ["a.py", "b.py"]) == "Add a score"


def test_a_project_tool_call_from_the_chat_model_is_understood_not_refused():
    from types import SimpleNamespace

    from ai.model_tool_calls import finish_model_response

    handled = []
    module = SimpleNamespace(
        engine=None,
        event_bus=SimpleNamespace(emit=lambda *a, **k: handled.append(("said", k))),
        _project_request_from_tool_call=lambda: handled.append("understood") or True,
    )
    assert finish_model_response(module, "<|tool_call_start|>[filesystem.list_files()]<|tool_call_end|>") is None
    assert handled == ["understood"]
