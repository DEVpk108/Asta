from __future__ import annotations

import copy
import sys
from pathlib import Path

from core import Kernel, Planner
from core.agent import AgentPlanProposal
from core.contracts import (
    IntentType,
    PlanStepStatus,
    TaskStatus,
    ToolRequest,
    ToolResult,
)
from core.intent_router import IntentRouter
from core.task_runtime import TaskRuntimeModule
from core.tools import (
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    RunCommandTool,
    ToolRuntimeModule,
    WriteWorkspaceFileTool,
)
from core.tools.sandbox import SandboxProcessResult


GOAL = (
    "Create main.py that prints exactly ASTA_OK. Run it, and tell me the result. "
    "Only work in this project."
)
FILE_CONTENT = 'print("ASTA_OK")\n'


class FakeSandboxRuntime:
    runtime_name = "test-container-runtime"

    def __init__(self, *, timeout_on_start=False):
        self.calls = []
        self.timeout_on_start = timeout_on_start
        self.snapshot_main = ""

    def run(self, arguments, *, timeout, output_limit):
        arguments = list(arguments)
        self.calls.append(arguments)
        if arguments[0] == "create":
            mount = arguments[arguments.index("--mount") + 1]
            source = mount.split("source=", 1)[1].split(",target=", 1)[0]
            main_file = Path(source) / "main.py"
            self.snapshot_main = (
                main_file.read_text(encoding="utf-8")
                if main_file.is_file()
                else ""
            )
            return SandboxProcessResult(0, stdout="test-container\n")
        if arguments[0] == "start":
            if self.timeout_on_start:
                return SandboxProcessResult(137, timed_out=True)
            stdout = "ASTA_OK\n" if "ASTA_OK" in self.snapshot_main else "WRONG\n"
            return SandboxProcessResult(0, stdout=stdout)
        return SandboxProcessResult(0)


class FixedPlanBrain:
    """Deterministic planner fixture; the live executor remains ASTA's runtime."""

    enabled = True

    def __init__(self, steps, replanned_steps=None):
        self.steps = copy.deepcopy(steps)
        self.replanned_steps = copy.deepcopy(replanned_steps)
        self.plan_calls = 0
        self.intents = []

    def plan(self, goal, *, intent):
        self.plan_calls += 1
        self.intents.append(copy.deepcopy(intent.entities))
        steps = (
            self.replanned_steps
            if self.plan_calls > 1 and self.replanned_steps is not None
            else self.steps
        )
        return AgentPlanProposal(
            goal_summary="Create and verify a tiny workspace program.",
            success_conditions=(
                "main.py contains the requested print statement",
                "running main.py exits successfully and prints ASTA_OK",
            ),
            rationale="Use the workspace file tools and verify observed results.",
            steps=tuple(copy.deepcopy(steps)),
            uncertainty=0.0,
        )

    @staticmethod
    def task_metadata(proposal):
        return {
            "agent_mode": "cognitive_v1",
            "agent_goal_summary": proposal.goal_summary,
            "agent_success_conditions": list(proposal.success_conditions),
            "agent_rationale": proposal.rationale,
            "agent_uncertainty": proposal.uncertainty,
        }


class RecordingEngine:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("A fixed acceptance plan must not call the chat model")


def _happy_plan():
    return [
        {
            "action": "list_files",
            "tool": "filesystem.list_files",
            "path": ".",
            "description": "Inspect the project workspace",
        },
        {
            "action": "write_file",
            "tool": "filesystem.write_file",
            "path": "main.py",
            "content": FILE_CONTENT,
            "description": "Create main.py",
        },
        {
            "action": "read_file",
            "tool": "filesystem.read_file",
            "path": "main.py",
            "expected_path": "main.py",
            "expected_content": FILE_CONTENT,
            "verification": "filesystem.file_content",
            "description": "Verify main.py contents",
        },
        {
            "action": "run",
            "tool": "system.run_command",
            "target": sys.executable,
            "arguments": ["main.py"],
            "cwd": ".",
            "verification": "command.output",
            "expected_returncode": 0,
            "expected_stdout": "ASTA_OK",
            "stdout_match": "exact",
            "description": "Run main.py and verify output",
        },
    ]


def _recovery_plan():
    return [
        {
            "action": "list_files",
            "tool": "filesystem.list_files",
            "path": ".",
            "description": "Inspect the project workspace",
        },
        {
            "action": "read_file",
            "tool": "filesystem.read_file",
            "path": "main.py",
            "description": "Read the existing file before correction",
        },
        {
            "action": "write_file",
            "tool": "filesystem.write_file",
            "path": "main.py",
            "content": FILE_CONTENT,
            "expected_sha256_from_step": "step-2",
            "description": "Correct main.py using the observed file revision",
        },
        {
            "action": "read_file",
            "tool": "filesystem.read_file",
            "path": "main.py",
            "expected_path": "main.py",
            "expected_content": FILE_CONTENT,
            "verification": "filesystem.file_content",
            "description": "Verify the corrected file",
        },
        {
            "action": "run",
            "tool": "system.run_command",
            "target": sys.executable,
            "arguments": ["main.py"],
            "cwd": ".",
            "verification": "command.output",
            "expected_returncode": 0,
            "expected_stdout": "ASTA_OK",
            "stdout_match": "exact",
            "description": "Run main.py and verify output",
        },
    ]


def _wrong_output_plan():
    return [
        {
            "action": "list_files",
            "tool": "filesystem.list_files",
            "path": ".",
            "description": "Inspect the project workspace",
        },
        {
            "action": "run",
            "tool": "system.run_command",
            "target": sys.executable,
            "arguments": ["main.py"],
            "cwd": ".",
            "verification": "command.output",
            "expected_returncode": 0,
            "expected_stdout": "ASTA_OK",
            "stdout_match": "exact",
            "description": "Run main.py and verify output",
        },
    ]


def _build_runtime(
    workspace,
    authority_path,
    plan,
    replanned_plan=None,
    *,
    authorize_run=True,
):
    kernel = Kernel(authority_path=authority_path)
    kernel.workspace_manager.update_project(
        name="acceptance-project",
        path=str(workspace),
    )
    kernel.register_tool(ListWorkspaceFilesTool(kernel.workspace_manager))
    kernel.register_tool(ReadWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(WriteWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(
        RunCommandTool(
            kernel.workspace_manager,
            runtime=FakeSandboxRuntime(),
        )
    )

    brain = FixedPlanBrain(plan, replanned_steps=replanned_plan)
    kernel.agent_brain = brain
    kernel.planner = Planner(kernel.tool_registry, agent_brain=brain)
    # This is test-only authorization for the known, generated one-line script.
    # Production risk settings remain unchanged: system.run_command is critical.
    if authorize_run:
        kernel.authority_manager.grant(
            "system.run_command",
            reason="isolated acceptance fixture",
        )

    tasks = TaskRuntimeModule(kernel)
    # Exercise the deterministic plan through the real task and tool runtimes.
    # No live model is needed to decide follow-up actions in this fixed-plan test.
    tasks._run_agent_decision = lambda *args, **kwargs: False
    tools = ToolRuntimeModule(kernel)
    from ai.ai_module import AIModule

    ai = AIModule(kernel)
    ai.engine = RecordingEngine()
    tasks.initialize()
    ai.initialize()
    tools.initialize()
    return kernel, tasks, ai, tools


def _shutdown(tasks, ai, tools):
    tools.shutdown()
    ai.shutdown()
    tasks.shutdown()


def _request(tool, arguments):
    return ToolRequest(tool=tool, arguments=arguments, request_id="workspace-test")


def _try_create_symlink(link, target):
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        # Windows may require Developer Mode or elevated privileges. The
        # non-symlink path and traversal checks still run in that environment.
        return False
    return True


def test_project_task_language_routes_to_bounded_task_intent():
    result = IntentRouter().analyze(GOAL)

    assert result.intent is IntentType.TASK
    assert result.entities["capability_scope"] == [
        "filesystem.list_files",
        "filesystem.read_file",
        "filesystem.write_file",
        "system.run_command",
    ]


def test_accepts_and_verifies_a_workspace_scoped_python_task(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(
        workspace,
        tmp_path / "authority.json",
        _happy_plan(),
    )
    messages = []
    kernel.event_bus.subscribe(
        "assistant_sentence",
        lambda text=None, **kwargs: messages.append(text),
    )
    try:
        kernel.event_bus.emit("user_message", GOAL)

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
        assert [step.metadata["tool"] for step in task.plan.steps] == [
            "filesystem.list_files",
            "filesystem.write_file",
            "filesystem.read_file",
            "system.run_command",
        ]
        assert all(step.status.value == "completed" for step in task.plan.steps)
        assert all(
            str((workspace / item["output"]["path"]).resolve()).startswith(
                str(workspace.resolve())
            )
            for item in task.evidence
            if item.get("tool") in {"filesystem.read_file", "filesystem.write_file"}
            and isinstance(item.get("output"), dict)
            and item["output"].get("path")
        )

        result = task.result
        assert result["verified"] is True
        assert result["artifacts"] == [{"path": "main.py"}]
        assert result["command"]["returncode"] == 0
        assert result["command"]["stdout"].strip() == "ASTA_OK"
        assert [item["status"] for item in result["verifications"]] == [
            "verified",
            "verified",
        ]
        assert messages
        assert "main.py" in messages[-1]
        assert "ASTA_OK" in messages[-1]
    finally:
        _shutdown(tasks, ai, tools)


def test_replans_after_wrong_output_then_corrects_and_verifies(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "main.py").write_text('print("WRONG")\n', encoding="utf-8")
    kernel, tasks, ai, tools = _build_runtime(
        workspace,
        tmp_path / "authority.json",
        _wrong_output_plan(),
        _recovery_plan(),
    )
    try:
        kernel.event_bus.emit("user_message", GOAL)

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
        initial_bad_result = next(
            item
            for item in task.evidence
            if item.get("tool") == "system.run_command"
            and isinstance(item.get("output"), dict)
            and str(item["output"].get("stdout") or "").strip() == "WRONG"
        )
        assert initial_bad_result["verification"]["status"] == "failed"
        assert task.metadata["replan_attempts"] == 1
        assert task.plan.steps[2].metadata["expected_sha256_from_step"] == "step-2"
        assert task.result["command"]["stdout"].strip() == "ASTA_OK"
        recovery_evidence = kernel.agent_brain.intents[1]["recovery_context"][
            "recent_evidence"
        ]
        assert any(
            item.get("output", {}).get("stdout") == "WRONG\n"
            for item in recovery_evidence
        )
    finally:
        _shutdown(tasks, ai, tools)


def test_workspace_file_tools_reject_traversal_absolute_paths_and_escaping_symlinks(
    tmp_path,
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch", encoding="utf-8")
    has_escape_symlink = _try_create_symlink(workspace / "escape.txt", outside)

    kernel = Kernel()
    kernel.workspace_manager.update_project(path=str(workspace))
    read = ReadWorkspaceFileTool(kernel.workspace_manager)
    write = WriteWorkspaceFileTool(kernel.workspace_manager)
    listing = ListWorkspaceFilesTool(kernel.workspace_manager)

    unsafe_read_paths = ["../outside.txt", str(outside), "C:\\outside.txt"]
    if has_escape_symlink:
        unsafe_read_paths.append("escape.txt")
    for path in unsafe_read_paths:
        result = read.execute(_request("filesystem.read_file", {"path": path}))
        assert not result.success

    unsafe_write_paths = ["../outside.txt", str(outside)]
    if has_escape_symlink:
        unsafe_write_paths.append("escape.txt")
    for path in unsafe_write_paths:
        result = write.execute(
            _request(
                "filesystem.write_file",
                {"path": path, "content": "changed"},
            )
        )
        assert not result.success

    listed = listing.execute(
        _request("filesystem.list_files", {"path": ".", "recursive": True})
    )
    assert listed.success
    if has_escape_symlink:
        assert all(item["path"] != "escape.txt" for item in listed.output["items"])
    assert outside.read_text(encoding="utf-8") == "do not touch"


def test_workspace_file_tools_hide_sensitive_aliases_and_allow_env_example(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "settings.txt").write_text("ordinary", encoding="utf-8")
    env_alias = workspace / ".env"
    if not _try_create_symlink(env_alias, workspace / "settings.txt"):
        # Exercise the same sensitive-name guard if the host forbids symlinks.
        env_alias.write_text("SECRET=value\n", encoding="utf-8")
    (workspace / ".env.example").write_text("TEMPLATE=value\n", encoding="utf-8")

    kernel = Kernel()
    kernel.workspace_manager.update_project(path=str(workspace))
    read = ReadWorkspaceFileTool(kernel.workspace_manager)
    listing = ListWorkspaceFilesTool(kernel.workspace_manager)

    assert not read.execute(
        _request("filesystem.read_file", {"path": ".env"})
    ).success
    assert read.execute(
        _request("filesystem.read_file", {"path": ".env.example"})
    ).success
    items = listing.execute(
        _request("filesystem.list_files", {"path": ".", "recursive": True})
    ).output["items"]
    paths = {item["path"] for item in items}
    assert ".env" not in paths
    assert ".env.example" in paths


def test_run_command_is_workspace_rooted_and_times_out(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel = Kernel()
    kernel.workspace_manager.update_project(path=str(workspace))
    runtime = FakeSandboxRuntime(timeout_on_start=True)
    tool = RunCommandTool(kernel.workspace_manager, runtime=runtime)

    escaped = tool.execute(
        ToolRequest(
            tool="system.run_command",
            arguments={
                "target": sys.executable,
                "arguments": ["-c", "print('should not run')"],
                "cwd": "../",
            },
            request_id="escape",
            timeout_seconds=0.2,
        )
    )
    assert not escaped.success

    timed_out = tool.execute(
        ToolRequest(
            tool="system.run_command",
            arguments={
                "target": sys.executable,
                "arguments": ["-c", "import time; time.sleep(2)"],
                "cwd": ".",
            },
            request_id="timeout",
            timeout_seconds=0.1,
        )
    )
    assert not timed_out.success
    assert "timed out" in timed_out.error.lower()
    assert any(call[0] == "kill" for call in runtime.calls)
    assert runtime.calls[-1][:2] == ["rm", "--force"]


def test_write_waits_for_explicit_authorization_before_touching_file(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(path=str(workspace))
    kernel.register_tool(WriteWorkspaceFileTool(kernel.workspace_manager))
    kernel.authority_manager.require_confirmation("filesystem.write_file")

    result = kernel.tool_dispatcher.dispatch(
        _request(
            "filesystem.write_file",
            {"path": "main.py", "content": FILE_CONTENT},
        )
    )

    assert not result.success
    assert result.metadata["requires_confirmation"] is True
    assert not (workspace / "main.py").exists()
    kernel.register_tool(RunCommandTool(kernel.workspace_manager))
    command_result = kernel.tool_dispatcher.dispatch(
        ToolRequest(
            tool="system.run_command",
            arguments={
                "target": sys.executable,
                "arguments": [
                    "-c",
                    "from pathlib import Path; Path('command-ran').write_text('yes')",
                ],
                "cwd": ".",
            },
            request_id="critical-command",
        )
    )
    assert not command_result.success
    assert command_result.metadata["requires_confirmation"] is True
    assert not (workspace / "command-ran").exists()


def test_cancellation_stops_later_tool_requests(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(path=str(workspace))
    kernel.register_tool(ListWorkspaceFilesTool(kernel.workspace_manager))
    kernel.register_tool(ReadWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(WriteWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(RunCommandTool(kernel.workspace_manager))
    brain = FixedPlanBrain(_happy_plan())
    kernel.agent_brain = brain
    kernel.planner = Planner(kernel.tool_registry, agent_brain=brain)
    tasks = TaskRuntimeModule(kernel)
    tasks.initialize()
    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request=None, **kwargs: requests.append(request),
    )
    try:
        intent = kernel.intent_router.analyze(GOAL)
        task = tasks.start_plan(GOAL, intent)
        assert task is not None
        kernel.task_manager.cancel("test cancellation", task.id)

        first = tasks._next_ready_plan_step(task)
        assert first is not None
        request = tasks.build_plan_request(task, first)
        kernel.event_bus.emit("tool_request", request=request)

        assert task.status is TaskStatus.CANCELLED
        assert requests == [request]
        assert not (workspace / "main.py").exists()
    finally:
        tasks.shutdown()


def test_successful_process_without_observable_stdout_stays_unverified(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(
        workspace,
        tmp_path / "authority.json",
        _happy_plan(),
    )

    original_dispatch = kernel.tool_dispatcher.dispatch

    def omit_command_stdout(request, *, confirmed=False):
        if request.tool == "system.run_command":
            return ToolResult(
                success=True,
                tool=request.tool,
                output={"returncode": 0},
            )
        return original_dispatch(request, confirmed=confirmed)

    kernel.tool_dispatcher.dispatch = omit_command_stdout
    try:
        kernel.event_bus.emit("user_message", GOAL)

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.PAUSED
        assert task.plan.steps[-1].status is PlanStepStatus.BLOCKED
        assert task.result is None
        assert task.evidence[-1]["verification"]["status"] == "unknown"
    finally:
        _shutdown(tasks, ai, tools)


def test_denied_critical_command_is_not_retried_or_executed(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(
        workspace,
        tmp_path / "authority.json",
        _happy_plan(),
        authorize_run=False,
    )

    class RecordingRunTool(RunCommandTool):
        def __init__(self, manager):
            super().__init__(manager)
            self.calls = 0

        def execute(self, request):
            self.calls += 1
            return super().execute(request)

    kernel.tool_registry.unregister("system.run_command")
    run_tool = RecordingRunTool(kernel.workspace_manager)
    kernel.register_tool(run_tool)
    confirmations = []
    kernel.event_bus.subscribe(
        "tool_confirmation_required",
        lambda request=None, **kwargs: confirmations.append(request),
    )
    try:
        kernel.event_bus.emit("user_message", GOAL)
        assert len(confirmations) == 1
        pending_request = confirmations[0]
        kernel.event_bus.emit(
            "tool_confirmation_response",
            request_id=pending_request.request_id,
            approved=False,
        )

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.FAILED
        assert run_tool.calls == 0
        assert not kernel.approval_manager.list_pending()
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
    finally:
        _shutdown(tasks, ai, tools)