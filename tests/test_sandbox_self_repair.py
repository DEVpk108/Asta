"""ASTA repairs its own code sandbox (Docker stopped / image missing) and resumes."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

from core import Kernel, Planner
from core.autonomy.capability_setup import CapabilitySetupManager
from core.autonomy.sandbox_setup import SandboxSetupOperator, SandboxSetupResult, SandboxSetupStatus
from core.contracts import TaskStatus
from core.task_runtime import TaskRuntimeModule
from core.tools import (
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    RunCommandTool,
    ToolRuntimeModule,
    WriteWorkspaceFileTool,
)
from core.tools.sandbox import SandboxProcessResult
from tests.test_project_edit_fix_loop import (
    BUGGY,
    FIXED,
    GOAL,
    LocalPythonSandbox,
    NoChat,
    SequencePlanBrain,
    _edit_plan,
    _project,
)

ENGINE_DOWN = (
    "failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; "
    "check if the path is correct and if the daemon is running"
)


# ----------------------------------------------------------------------------
# The operator: observe -> fix -> verify
# ----------------------------------------------------------------------------
class FakeDocker:
    def __init__(self, *, engine=False, image=True, boots_after=2, build_ok=True):
        self.engine = engine
        self.image = image
        self.boots_after = boots_after
        self.build_ok = build_ok
        self.launched = []
        self.calls = []
        self.polls = 0

    def run(self, command, *, timeout, output_limit):
        args = command[1:]
        self.calls.append(args)
        if args[0] == "info":
            if not self.engine and self.launched:
                self.polls += 1
                self.engine = self.polls >= self.boots_after
            return SandboxProcessResult(0 if self.engine else 1, stdout="27.0\n" if self.engine else "")
        if args[:2] == ["image", "inspect"]:
            return SandboxProcessResult(0 if self.image else 1)
        if args[0] == "build":
            self.image = self.build_ok
            return SandboxProcessResult(0 if self.build_ok else 1, stderr="" if self.build_ok else "network unreachable")
        return SandboxProcessResult(0)

    def launch(self, command):
        self.launched.append(command)


def _operator(docker, tmp_path, monkeypatch, *, cli=True, notes=None, timeout="30"):
    desktop = tmp_path / "Docker Desktop.exe"
    desktop.write_bytes(b"")
    monkeypatch.setenv("ASTA_DOCKER_DESKTOP_PATH", str(desktop))
    monkeypatch.setenv("ASTA_SANDBOX_STARTUP_TIMEOUT", timeout)
    repo = tmp_path / "repo"
    (repo / "sandbox").mkdir(parents=True)
    (repo / "sandbox" / "Dockerfile").write_bytes(b"FROM python:3.12-slim\n")
    clock = SimpleNamespace(now=0.0)
    return SandboxSetupOperator(
        notify=(notes.append if notes is not None else None),
        runtime_name="docker",
        resolve=lambda name: "C:/docker.exe" if cli else None,
        run=docker.run,
        launch=docker.launch,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        clock=lambda: clock.now,
        platform="win32",
        repo_root=repo,
    )


def test_starts_docker_desktop_waits_for_it_and_reports_what_it_did(tmp_path, monkeypatch):
    docker = FakeDocker(engine=False, image=True)
    notes = []
    result = _operator(docker, tmp_path, monkeypatch, notes=notes).run({"error": ENGINE_DOWN})

    assert result.status is SandboxSetupStatus.COMPLETED
    assert docker.launched == [[str(tmp_path / "Docker Desktop.exe")]]
    assert result.metadata["actions"] == ["started Docker Desktop"]
    assert "docker engine is not running" in result.metadata["observed"]
    assert "Docker Desktop, which isn't running. I'm starting it" in notes[0]
    assert "started Docker Desktop" in result.summary


def test_gives_up_when_docker_never_becomes_ready(tmp_path, monkeypatch):
    docker = FakeDocker(engine=False, boots_after=10_000)
    result = _operator(docker, tmp_path, monkeypatch, timeout="12").run()

    assert result.status is SandboxSetupStatus.FAILED
    assert "wasn't ready after 12 seconds" in result.summary
    assert docker.polls <= 5  # bounded polling, not a hang


def test_missing_docker_cli_is_explained_not_guessed(tmp_path, monkeypatch):
    result = _operator(FakeDocker(), tmp_path, monkeypatch, cli=False).run()

    assert result.status is SandboxSetupStatus.FAILED
    assert result.metadata["fixable"] is False
    assert "isn't installed" in result.summary


def test_builds_the_missing_image_from_astas_own_dockerfile(tmp_path, monkeypatch):
    docker = FakeDocker(engine=True, image=False)
    notes = []
    result = _operator(docker, tmp_path, monkeypatch, notes=notes).run()

    assert result.status is SandboxSetupStatus.COMPLETED
    [build] = [call for call in docker.calls if call[0] == "build"]
    assert build[:4] == ["build", "--pull", "-t", "asta-python-sandbox:1"]
    assert build[-2].endswith(str(Path("sandbox") / "Dockerfile"))
    assert any("building it once" in note for note in notes)


def test_failed_build_and_disabled_auto_build_say_what_to_run(tmp_path, monkeypatch):
    failed = _operator(FakeDocker(engine=True, image=False, build_ok=False), tmp_path, monkeypatch).run()
    assert failed.status is SandboxSetupStatus.FAILED and "network unreachable" in failed.summary

    monkeypatch.setenv("ASTA_SANDBOX_AUTO_BUILD", "0")
    (tmp_path / "b").mkdir()
    manual = _operator(FakeDocker(engine=True, image=False), tmp_path / "b", monkeypatch).run()
    assert manual.status is SandboxSetupStatus.FAILED
    assert "docker build --pull -t asta-python-sandbox:1" in manual.summary


def test_healthy_sandbox_needs_no_action(tmp_path, monkeypatch):
    docker = FakeDocker(engine=True, image=True)
    result = _operator(docker, tmp_path, monkeypatch).run()

    assert result.status is SandboxSetupStatus.COMPLETED
    assert docker.launched == [] and result.metadata["actions"] == []


# ----------------------------------------------------------------------------
# End to end: approve once -> Docker down -> repaired -> same command reruns
# ----------------------------------------------------------------------------
class StoppedThenStartedSandbox(LocalPythonSandbox):
    def __init__(self):
        super().__init__()
        self.engine_running = False

    def run(self, arguments, *, timeout, output_limit):
        if arguments[0] == "create" and not self.engine_running:
            return SandboxProcessResult(1, stderr=ENGINE_DOWN)
        return super().run(arguments, timeout=timeout, output_limit=output_limit)


def _runtime(tmp_path, workspace, sandbox, operator):
    import os

    os.environ["ASTA_SANDBOX_SELF_REPAIR"] = "1"  # conftest resets it per test
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    kernel.register_tool(ListWorkspaceFilesTool(kernel.workspace_manager))
    kernel.register_tool(ReadWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(WriteWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(RunCommandTool(kernel.workspace_manager, runtime=sandbox))
    kernel.capability_setup_manager = CapabilitySetupManager(
        kernel.media_manager, event_bus=kernel.event_bus, operators={"sandbox": operator},
    )
    brain = SequencePlanBrain(_edit_plan(FIXED))
    kernel.agent_brain = brain
    kernel.planner = Planner(kernel.tool_registry, agent_brain=brain)
    tasks = TaskRuntimeModule(kernel)
    tasks._run_agent_decision = lambda *args, **kwargs: False
    tools = ToolRuntimeModule(kernel)
    from ai.ai_module import AIModule

    ai = AIModule(kernel)
    ai.engine = NoChat()
    for module in (tasks, ai, tools):
        module.initialize()
    spoken, confirmations = [], []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    kernel.event_bus.subscribe("tool_confirmation_required", lambda request=None, **kw: confirmations.append(request))
    return SimpleNamespace(kernel=kernel, spoken=spoken, confirmations=confirmations, modules=(tools, ai, tasks))


def _wait_terminal(kernel, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        task = kernel.task_manager.list()[0]
        if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return task
        time.sleep(0.02)
    return kernel.task_manager.list()[0]


def test_docker_down_is_repaired_and_the_approved_command_reruns_once(tmp_path):
    workspace = _project(tmp_path)
    sandbox = StoppedThenStartedSandbox()

    def start_docker(context):
        assert "docker API" in context["error"]
        sandbox.engine_running = True
        return SandboxSetupResult(
            SandboxSetupStatus.COMPLETED,
            "The code sandbox is ready (started Docker Desktop). Continuing your task.",
            {"observed": ["docker engine is not running"], "actions": ["started Docker Desktop"]},
        )

    run = _runtime(tmp_path, workspace, sandbox, start_docker)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        assert len(run.confirmations) == 1  # the user is asked once
        run.kernel.event_bus.emit(
            "tool_confirmation_response", request_id=run.confirmations[0].request_id, approved=True,
        )

        task = _wait_terminal(run.kernel)
        assert task.status is TaskStatus.COMPLETED, task.error
        assert len(run.confirmations) == 1  # ...and not again after the repair
        assert sandbox.runs and sandbox.runs[-1].returncode == 0
        assert (workspace / "calc.py").read_text(encoding="utf-8") == FIXED
        assert any("started Docker Desktop" in str(text) for text in run.spoken)
        assert "ALL TESTS PASSED" in run.spoken[-1]
        assert not any("I paused the project task" in str(text) for text in run.spoken)
        repair = next(item for item in task.evidence if item.get("type") == "capability_setup")
        assert repair["actions"] == ["started Docker Desktop"]
    finally:
        for module in run.modules:
            module.shutdown()


def test_failed_repair_fails_the_task_with_the_reason(tmp_path):
    workspace = _project(tmp_path)
    reason = "I can't create the code sandbox because Docker Desktop isn't installed (or isn't on PATH)."
    run = _runtime(
        tmp_path, workspace, StoppedThenStartedSandbox(),
        lambda context: SandboxSetupResult(SandboxSetupStatus.FAILED, reason, {"fixable": False}),
    )
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        run.kernel.event_bus.emit(
            "tool_confirmation_response", request_id=run.confirmations[0].request_id, approved=True,
        )

        task = _wait_terminal(run.kernel)
        assert task.status is TaskStatus.FAILED
        assert task.error == reason
        assert (workspace / "calc.py").read_text(encoding="utf-8") == FIXED  # the edit itself stays
    finally:
        for module in run.modules:
            module.shutdown()


def test_repair_is_tried_once_per_task(tmp_path):
    workspace = _project(tmp_path)
    calls = []

    def claims_success(context):
        calls.append(context)  # says fixed, but the engine stays down
        return SandboxSetupResult(SandboxSetupStatus.COMPLETED, "The code sandbox is ready.", {})

    run = _runtime(tmp_path, workspace, StoppedThenStartedSandbox(), claims_success)
    try:
        run.kernel.event_bus.emit("user_message", GOAL)
        run.kernel.event_bus.emit(
            "tool_confirmation_response", request_id=run.confirmations[0].request_id, approved=True,
        )

        task = _wait_terminal(run.kernel)
        assert task.status is TaskStatus.FAILED
        assert len(calls) == 1
        assert len(run.confirmations) == 1
    finally:
        for module in run.modules:
            module.shutdown()
