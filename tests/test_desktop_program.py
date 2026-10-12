"""Write a game and open it on the desktop; everyday commands beside a task."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from core.autonomy.verification import VerificationEngine
from core.background_tasks import planning_failure_reason
from core.contracts import TaskStatus, ToolDefinition, ToolRequest, ToolResult
from core.phrasing import LINES
from core.planner import Planner
from core.tools import OpenProjectProgramTool
from core.tools.base import Tool
from tests.test_project_edit_fix_loop import _runtime, _shutdown

GAME = "import tkinter as tk\nroot = tk.Tk()\nroot.title('Snake')\nroot.mainloop()\n"
GAME_V2 = "import tkinter as tk\nroot = tk.Tk()\nroot.title('Snake v2')\nroot.mainloop()\n"
GOAL = "Write code for a Snake game in Python. After that run it."


class FakeProcess:
    pid = 4242

    def __init__(self, returncode, stderr_text, stderr_file):
        self._returncode = returncode
        stderr_file.write(stderr_text.encode("utf-8"))
        stderr_file.flush()

    def poll(self):
        return self._returncode


class FakePopen:
    """Records launches; the n-th launch crashes or keeps running."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes) or [(None, "")]
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        returncode, stderr = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        return FakeProcess(returncode, stderr, kwargs["stderr"])


def _game_plan(content):
    return [
        {"action": "write_file", "tool": "filesystem.write_file", "path": "snake.py", "content": content},
        # No expected_content: ASTA copies the written text into the check.
        {"action": "read_file", "tool": "filesystem.read_file", "path": "snake.py",
         "verification": "filesystem.file_content"},
        {"action": "open_program", "tool": "project.open_program", "path": "snake.py"},
    ]


def _fix_plan(content):
    return [
        {"action": "read_file", "tool": "filesystem.read_file", "path": "snake.py"},
        {"action": "write_file", "tool": "filesystem.write_file", "path": "snake.py", "content": content,
         "expected_sha256_from_step": "step-1"},
        {"action": "read_file", "tool": "filesystem.read_file", "path": "snake.py",
         "verification": "filesystem.file_content"},
        {"action": "open_program", "tool": "project.open_program", "path": "snake.py"},
    ]


def _empty_project(tmp_path):
    workspace = tmp_path / "games"
    workspace.mkdir()
    return workspace


def _with_program_tool(run, popen):
    tool = OpenProjectProgramTool(run.kernel.workspace_manager, popen=popen, settle_seconds=0.2)
    run.kernel.register_tool(tool)
    return tool


def _approve_all(run, deadline=10.0):
    end = time.monotonic() + deadline
    answered = set()
    while time.monotonic() < end:
        for request in list(run.confirmations):
            if request.request_id not in answered:
                answered.add(request.request_id)
                run.kernel.event_bus.emit("tool_confirmation_response", request_id=request.request_id, approved=True)
        task = run.kernel.task_manager.list()[0] if run.kernel.task_manager.list() else None
        if task is not None and task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return task
        time.sleep(0.02)
    return run.kernel.task_manager.list()[0]


# --- planning ------------------------------------------------------------------

def test_readback_content_is_copied_and_a_missing_readback_is_added():
    steps = Planner._complete_project_commands([
        {"action": "read_file", "path": "a.py"},
        {"action": "write_file", "path": "a.py", "content": "A\n", "expected_sha256_from_step": "step-1"},
        {"action": "write_file", "path": "b.py", "content": "B\n"},
        {"action": "read_file", "path": "b.py", "verification": "filesystem.file_content"},
        {"action": "read_file", "path": "c.py"},
        {"action": "write_file", "path": "c.py", "content": "C\n", "expected_sha256_from_step": "step-5"},
        {"action": "open_program", "path": "b.py", "expected_stdout": "x"},
    ])
    assert [s["action"] for s in steps] == [
        "read_file", "write_file", "read_file", "write_file", "read_file", "read_file", "write_file", "read_file",
        "open_program",
    ]
    assert steps[2]["expected_content"] == "A\n" and steps[2]["verification"] == "filesystem.file_content"
    assert steps[4]["expected_content"] == "B\n" and steps[4]["expected_path"] == "b.py"
    assert steps[6]["expected_sha256_from_step"] == "step-6"  # shifted by the inserted read-back
    assert steps[7]["expected_content"] == "C\n"
    assert steps[8]["verification"] == "program.opened" and "expected_stdout" not in steps[8]


def test_cut_off_plan_says_why():
    reason = planning_failure_reason(
        "Cognitive planning failed: ModelRoutingError: Agent generation incomplete (output limit); no partial plan will execute."
    )
    assert "ASTA_CODING_AGENT_MAX_OUTPUT_TOKENS" in reason


# --- the tool ------------------------------------------------------------------

def _workspace_manager(root):
    from core import Kernel

    kernel = Kernel(authority_path=root / "authority.json")
    project = root / "proj"
    project.mkdir()
    kernel.workspace_manager.update_project(name="p", path=str(project))
    return kernel.workspace_manager, project


def _request(path, **arguments):
    return ToolRequest(tool="project.open_program", arguments={"path": path, **arguments}, request_id="r1")


def test_program_that_keeps_running_counts_as_opened(tmp_path):
    manager, project = _workspace_manager(tmp_path)
    (project / "snake.py").write_bytes(GAME.encode())
    popen = FakePopen((None, ""))
    result = OpenProjectProgramTool(manager, popen=popen, settle_seconds=0.2).execute(_request("snake.py"))
    assert result.success, result.error
    assert result.output["state"] == "running"
    argv, kwargs = popen.calls[0]
    assert argv[1:] == ["snake.py"] and Path(kwargs["cwd"]) == project
    step = SimpleNamespace(id="s", metadata={"verification": "program.opened"})
    assert VerificationEngine().verify(None, step, result).verified


def test_a_crash_on_start_is_a_failure_with_the_error(tmp_path):
    manager, project = _workspace_manager(tmp_path)
    (project / "snake.py").write_bytes(b"import pygame\n")
    popen = FakePopen((1, "Traceback...\nModuleNotFoundError: No module named 'pygame'\n"))
    result = OpenProjectProgramTool(manager, popen=popen, settle_seconds=0.2).execute(_request("snake.py"))
    assert not result.success
    assert "No module named 'pygame'" in result.error
    assert "pygame" in result.output["stderr"]


def test_only_workspace_python_files_can_be_opened(tmp_path):
    manager, project = _workspace_manager(tmp_path)
    (project / "notes.txt").write_bytes(b"hi")
    popen = FakePopen()
    tool = OpenProjectProgramTool(manager, popen=popen, settle_seconds=0.2)
    for path in ("notes.txt", "../outside.py", "C:/Windows/evil.py", "missing.py"):
        assert not tool.execute(_request(path)).success, path
    assert popen.calls == []
    assert tool.definition.risk_level == "critical"


def test_project_venv_python_is_used(tmp_path):
    manager, project = _workspace_manager(tmp_path)
    (project / "snake.py").write_bytes(GAME.encode())
    venv_python = project / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"")
    popen = FakePopen()
    OpenProjectProgramTool(manager, popen=popen, settle_seconds=0.2).execute(_request("snake.py"))
    assert popen.calls[0][0][0] == str(venv_python)


# --- end to end ----------------------------------------------------------------

def test_write_a_game_and_open_it_after_asking(tmp_path):
    workspace = _empty_project(tmp_path)
    run = _runtime(tmp_path, workspace, _game_plan(GAME))
    popen = FakePopen((None, ""))
    _with_program_tool(run, popen)
    heard = []
    run.kernel.event_bus.subscribe(
        "assistant_sentence", lambda text=None, spoken=None, **kw: heard.append(text if spoken is None else spoken)
    )
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        task = _approve_all(run)
        assert task.status is TaskStatus.COMPLETED, task.error
    finally:
        _shutdown(run)
    assert (workspace / "snake.py").read_bytes().decode() == GAME
    assert len(popen.calls) == 1
    assert [r.tool for r in run.confirmations] == ["project.open_program"]  # asked before opening
    assert any("on your desktop" in line for line in run.spoken), run.spoken
    assert heard[-1] in LINES["done_opened"], heard


def test_a_game_that_crashes_on_start_is_fixed_and_reopened(tmp_path):
    workspace = _empty_project(tmp_path)
    run = _runtime(tmp_path, workspace, _game_plan(GAME), _fix_plan(GAME_V2))
    popen = FakePopen((1, "ModuleNotFoundError: No module named 'pygame'\n"), (None, ""))
    _with_program_tool(run, popen)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        task = _approve_all(run)
        assert task.status is TaskStatus.COMPLETED, task.error
    finally:
        _shutdown(run)
    assert (workspace / "snake.py").read_bytes().decode() == GAME_V2
    assert len(popen.calls) == 2
    recovery = run.brain.intents[1].get("recovery_context")
    assert "pygame" in str(recovery)


# --- everyday commands next to a project task ----------------------------------

class _Fake(Tool):
    def __init__(self, name, actions, properties, required, calls):
        self._definition = ToolDefinition(
            name=name, description=name,
            input_schema={"type": "object", "properties": properties, "required": required},
            risk_level="low", metadata={"actions": actions},
        )
        self.calls = calls

    @property
    def definition(self):
        return self._definition

    def execute(self, request):
        self.calls.append((request.tool, dict(request.arguments)))
        output = {"target": request.arguments.get("target")}
        if request.tool == "media.ui_play":
            output = {"message": "Playing Baithi Hai on Spotify.", "verified": True}
        return ToolResult(success=True, tool=request.tool, output=output, metadata={"request_id": request.request_id})


def test_music_plays_while_a_project_task_runs(tmp_path):
    from core import Kernel
    from core.side_actions import SideActionRunner
    from core.contracts import IntentResult, IntentType

    kernel = Kernel(authority_path=tmp_path / "authority.json")
    calls = []
    kernel.register_tool(_Fake("system.open_application", ["open"], {"target": {"type": "string"}}, ["target"], calls))
    kernel.register_tool(_Fake("computer.wait", ["wait"], {"seconds": {"type": "number"}}, [], calls))
    kernel.register_tool(_Fake("media.ui_play", ["ui_play"], {"application": {"type": "string"},
                                                              "query": {"type": "string"}}, [], calls))
    kernel.task_runtime = SimpleNamespace(build_plan_request=None)
    from core.task_runtime import TaskRuntimeModule

    kernel.task_runtime = TaskRuntimeModule(kernel)
    said = []
    runner = SideActionRunner(kernel, say=said.append)
    plan_steps = [
        {"action": "open", "target": "spotify"},
        {"action": "wait", "seconds": 0.0},
        {"action": "ui_play", "application": "Spotify", "query": "Baithi Hai"},
    ]
    from core.contracts import Plan, PlanStep

    plan = Plan(goal="play", steps=[
        PlanStep(id=f"step-{i + 1}", description=s["action"], metadata={**s, "tool": t})
        for i, (s, t) in enumerate(zip(plan_steps, ["system.open_application", "computer.wait", "media.ui_play"]))
    ])
    runner._plan = lambda text, intent: plan
    intent = IntentResult(intent=IntentType.COMMAND, confidence=0.9, normalized_text="play baithi hai on spotify",
                          entities={"action": "media"}, requires_tools=True, classifier="rules")
    assert runner.try_start("Play baithi hai on spotify", intent)
    runner._thread.join(timeout=5)
    assert [c[0] for c in calls] == ["system.open_application", "computer.wait", "media.ui_play"]
    assert said == ["Playing Baithi Hai on Spotify."]


def test_project_tools_never_run_as_a_side_action(tmp_path):
    from core import Kernel
    from core.contracts import IntentResult, IntentType, Plan, PlanStep
    from core.side_actions import SideActionRunner

    kernel = Kernel(authority_path=tmp_path / "authority.json")
    runner = SideActionRunner(kernel, say=lambda text: None)
    runner._plan = lambda text, intent: Plan(goal="x", steps=[
        PlanStep(id="step-1", description="run", metadata={"action": "run", "tool": "system.run_command"}),
    ])
    intent = IntentResult(intent=IntentType.COMMAND, confidence=0.9, normalized_text="run main.py",
                          entities={"action": "run"}, requires_tools=True, classifier="rules")
    assert runner.try_start("run main.py", intent) is False
