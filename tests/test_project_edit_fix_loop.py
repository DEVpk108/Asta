"""Stage 1 acceptance: edit an existing file, run its test, fix a failure.

The plans come from a fixed planner fixture, but everything else is ASTA's
real runtime: task runtime, tool runtime, workspace tools, verification,
recovery/replanning and the sandboxed command tool. The fake sandbox runs the
command for real (with this Python) inside the read-only workspace snapshot.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import Kernel, Planner
from core.agent import AgentBrain, AgentPlanProposal
from core.contracts import IntentResult, IntentType, TaskStatus, ToolRequest
from core.planner import PlanningError
from core.task_runtime import TaskRuntimeModule
from core.tools import (
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    RunCommandTool,
    ToolRuntimeModule,
    WriteWorkspaceFileTool,
)
from core.tools.sandbox import SandboxProcessResult
from core.tools.workspace import planning_file_context, text_change_summary


GOAL = "Fix the bug in calc.py so the tests in test_calc.py pass."
BUGGY = "def add(a, b):\n    return a - b\n\n\ndef double(x):\n    return x * 2\n"
WRONG_FIX = "def add(a, b):\n    return a + b + 1\n\n\ndef double(x):\n    return x * 2\n"
FIXED = "def add(a, b):\n    return a + b\n\n\ndef double(x):\n    return x * 2\n"
TESTS = (
    "from calc import add, double\n\n"
    "assert add(2, 3) == 5, f'add(2, 3) returned {add(2, 3)}'\n"
    "assert double(4) == 8\n"
    "print('ALL TESTS PASSED')\n"
)
PROJECT_SCOPE = [
    "filesystem.list_files",
    "filesystem.read_file",
    "filesystem.write_file",
    "system.run_command",
]


class LocalPythonSandbox:
    """Runs the requested Python command for real inside the snapshot copy."""

    runtime_name = "test-local-python"

    def __init__(self):
        self.pending = None
        self.runs = []

    def run(self, arguments, *, timeout, output_limit):
        arguments = list(arguments)
        if arguments[0] == "create":
            mount = arguments[arguments.index("--mount") + 1]
            source = mount.split("source=", 1)[1].split(",target=", 1)[0]
            entry = arguments.index("--entrypoint")
            command = arguments[entry + 3:]
            self.pending = (Path(source), command)
            return SandboxProcessResult(0, stdout="container\n")
        if arguments[0] == "start":
            source, command = self.pending
            completed = subprocess.run(
                [sys.executable, *command],
                cwd=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.runs.append(completed)
            return SandboxProcessResult(
                completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        return SandboxProcessResult(0)


class SequencePlanBrain:
    """Returns the next fixed plan on every (re)plan call."""

    enabled = True

    def __init__(self, *plans):
        self.plans = [copy.deepcopy(plan) for plan in plans]
        self.intents = []
        self.goals = []

    def plan(self, goal, *, intent):
        self.goals.append(goal)
        self.intents.append(copy.deepcopy(intent.entities))
        steps = self.plans[min(len(self.intents), len(self.plans)) - 1]
        return AgentPlanProposal(
            goal_summary="Fix calc.py so its tests pass.",
            success_conditions=("python test_calc.py exits 0 and prints ALL TESTS PASSED",),
            rationale="Read the code, fix it, verify the file, run the tests.",
            steps=tuple(copy.deepcopy(steps)),
            uncertainty=0.1,
        )

    @staticmethod
    def task_metadata(proposal):
        return {"agent_mode": "cognitive_v1"}


class NoChat:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("fixed plans must not call the chat model")


def _run_tests_step():
    return {
        "action": "run",
        "tool": "system.run_command",
        "target": sys.executable,
        "arguments": ["test_calc.py"],
        "cwd": ".",
        "verification": "command.output",
        "expected_returncode": 0,
        "expected_stdout": "ALL TESTS PASSED",
        "stdout_match": "contains",
        "description": "Run the tests",
    }


def _check_first_plan():
    return [
        {"action": "list_files", "tool": "filesystem.list_files", "path": ".",
         "description": "Inspect the project"},
        _run_tests_step(),
    ]


def _edit_plan(content, *, path="calc.py"):
    return [
        {"action": "read_file", "tool": "filesystem.read_file", "path": path,
         "description": f"Read {path}"},
        {"action": "write_file", "tool": "filesystem.write_file", "path": path,
         "content": content, "expected_sha256_from_step": "step-1",
         "description": f"Fix {path}"},
        {"action": "read_file", "tool": "filesystem.read_file", "path": path,
         "verification": "filesystem.file_content", "expected_path": path,
         "expected_content": content, "description": f"Verify {path}"},
        _run_tests_step(),
    ]


def _project(tmp_path):
    # Bytes, not write_text: Windows would turn "\n" into "\r\n".
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "calc.py").write_bytes((BUGGY).encode("utf-8"))
    (workspace / "test_calc.py").write_bytes((TESTS).encode("utf-8"))
    return workspace


def _runtime(tmp_path, workspace, *plans):
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    kernel.register_tool(ListWorkspaceFilesTool(kernel.workspace_manager))
    kernel.register_tool(ReadWorkspaceFileTool(kernel.workspace_manager))
    kernel.register_tool(WriteWorkspaceFileTool(kernel.workspace_manager))
    sandbox = LocalPythonSandbox()
    kernel.register_tool(RunCommandTool(kernel.workspace_manager, runtime=sandbox))
    brain = SequencePlanBrain(*plans)
    kernel.agent_brain = brain
    kernel.planner = Planner(kernel.tool_registry, agent_brain=brain)
    # Test-only grant for the known test command; production keeps asking.
    kernel.authority_manager.grant("system.run_command", reason="isolated acceptance fixture")

    tasks = TaskRuntimeModule(kernel)
    tasks._run_agent_decision = lambda *args, **kwargs: False
    tools = ToolRuntimeModule(kernel)
    from ai.ai_module import AIModule

    ai = AIModule(kernel)
    ai.engine = NoChat()
    tasks.initialize()
    ai.initialize()
    tools.initialize()
    spoken = []
    confirmations = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    kernel.event_bus.subscribe(
        "tool_confirmation_required",
        lambda request=None, **kw: confirmations.append(request),
    )
    runtime = SimpleNamespace(
        kernel=kernel, brain=brain, sandbox=sandbox, spoken=spoken,
        confirmations=confirmations, modules=(tools, ai, tasks),
    )
    return runtime


def _shutdown(runtime):
    for module in runtime.modules:
        module.shutdown()


def test_failing_test_is_fixed_from_its_output_and_the_change_is_reported(tmp_path):
    workspace = _project(tmp_path)
    run = _runtime(tmp_path, workspace, _check_first_plan(), _edit_plan(FIXED))
    try:
        run.kernel.event_bus.emit("user_message", GOAL)

        task = run.kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED, task.error
        assert (workspace / "calc.py").read_text(encoding="utf-8") == FIXED
        assert (workspace / "test_calc.py").read_text(encoding="utf-8") == TESTS
        assert run.confirmations == []  # the user asked to fix calc.py

        # The first run really failed, and the replanner saw why.
        assert run.sandbox.runs[0].returncode != 0
        assert run.sandbox.runs[-1].returncode == 0
        recovery = run.brain.intents[1]["recovery_context"]["recent_evidence"]
        assert any("add(2, 3) returned -1" in str(item.get("output", {}).get("stderr")) for item in recovery)
        assert task.metadata["replan_attempts"] == 1

        result = task.result
        assert result["verified"] is True
        assert result["attempts"] == 2
        [change] = result["changes"]
        assert change["path"] == "calc.py" and change["created"] is False
        assert (change["lines_added"], change["lines_removed"]) == (1, 1)
        assert "-    return a - b" in change["diff"] and "+    return a + b" in change["diff"]
        message = run.spoken[-1]
        assert "Updated `calc.py` (+1 -1 lines)" in message
        assert "ALL TESTS PASSED" in message
        assert "It took 2 attempts" in message
    finally:
        _shutdown(run)


def test_second_fix_attempt_reports_one_net_change(tmp_path):
    workspace = _project(tmp_path)
    run = _runtime(
        tmp_path, workspace,
        _check_first_plan(), _edit_plan(WRONG_FIX), _edit_plan(FIXED),
    )
    try:
        run.kernel.event_bus.emit("user_message", GOAL)

        task = run.kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED, task.error
        assert (workspace / "calc.py").read_text(encoding="utf-8") == FIXED
        assert task.metadata["replan_attempts"] == 2
        # The second replan saw the wrong fix's failure and the current file.
        latest = run.brain.intents[2]["recovery_context"]["recent_evidence"]
        assert any("returned 6" in str(item.get("output", {}).get("stderr")) for item in latest)
        [change] = task.result["changes"]
        assert change["writes"] == 2
        assert (change["lines_added"], change["lines_removed"]) == (1, 1)
        assert "+ 1" not in change["diff"]
        assert "It took 3 attempts" in run.spoken[-1]
    finally:
        _shutdown(run)


def test_fix_loop_stops_after_its_retry_budget(tmp_path):
    workspace = _project(tmp_path)
    run = _runtime(tmp_path, workspace, _check_first_plan(), _edit_plan(WRONG_FIX))
    try:
        run.kernel.event_bus.emit("user_message", GOAL)

        task = run.kernel.task_manager.list()[0]
        assert task.status is TaskStatus.FAILED
        assert "exhausted" in str(task.error)
        assert len(run.brain.intents) == 4  # first plan + 3 bounded repairs
        assert all(item.returncode != 0 for item in run.sandbox.runs)
    finally:
        _shutdown(run)


def test_rewriting_an_unrequested_test_file_needs_approval(tmp_path):
    workspace = _project(tmp_path)
    loosened = TESTS.replace("== 5", "== -1")
    run = _runtime(tmp_path, workspace, _check_first_plan(), _edit_plan(loosened, path="test_calc.py"))
    try:
        run.kernel.event_bus.emit("user_message", "Fix calc.py so the tests pass.")

        assert len(run.confirmations) == 1
        request = run.confirmations[0]
        assert request.metadata["confirm_kind"] == "test_edit"
        assert request.metadata["confirm_overwrite"] == "test_calc.py"
        assert any("test file test_calc.py" in str(text) for text in run.spoken)
        assert (workspace / "test_calc.py").read_text(encoding="utf-8") == TESTS

        run.kernel.event_bus.emit(
            "tool_confirmation_response", request_id=request.request_id, approved=False,
        )
        task = run.kernel.task_manager.list()[0]
        assert task.status is TaskStatus.FAILED
        assert (workspace / "test_calc.py").read_text(encoding="utf-8") == TESTS
    finally:
        _shutdown(run)


def test_named_test_file_can_be_edited_without_extra_approval(tmp_path):
    workspace = _project(tmp_path)
    (workspace / "calc.py").write_bytes((FIXED).encode("utf-8"))
    broken_tests = TESTS.replace("== 8", "== 9")
    (workspace / "test_calc.py").write_bytes((broken_tests).encode("utf-8"))
    run = _runtime(tmp_path, workspace, _check_first_plan(), _edit_plan(TESTS, path="test_calc.py"))
    try:
        run.kernel.event_bus.emit("user_message", "Fix test_calc.py, its double() assertion is wrong.")

        task = run.kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED, task.error
        assert run.confirmations == []
        assert (workspace / "test_calc.py").read_text(encoding="utf-8") == TESTS
    finally:
        _shutdown(run)


def test_write_tool_reports_the_line_change(tmp_path):
    workspace = _project(tmp_path)
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    read = ReadWorkspaceFileTool(kernel.workspace_manager).execute(
        ToolRequest(tool="filesystem.read_file", arguments={"path": "calc.py"}, request_id="r")
    )
    write = WriteWorkspaceFileTool(kernel.workspace_manager).execute(
        ToolRequest(
            tool="filesystem.write_file",
            arguments={"path": "calc.py", "content": FIXED, "expected_sha256": read.output["sha256"]},
            request_id="w",
        )
    )
    assert write.success, write.error
    assert write.output["previous_sha256"] == read.output["sha256"]
    assert (write.output["lines_added"], write.output["lines_removed"]) == (1, 1)
    assert write.output["diff"].startswith("--- a/calc.py\n+++ b/calc.py\n")


def test_change_summary_ignores_line_ending_only_differences():
    summary = text_change_summary("a\r\nb\r\n", "a\nb\n", "x.py")
    assert summary["lines_added"] == summary["lines_removed"] == 0 and summary["diff"] == ""


def test_planning_context_includes_named_files_and_their_tests_read_only(tmp_path):
    workspace = _project(tmp_path)
    (workspace / ".env").write_bytes(("TOKEN=secret\n").encode("utf-8"))
    (workspace / "notes.md").write_bytes(("notes\n").encode("utf-8"))
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))

    context = planning_file_context(kernel.workspace_manager, "fix the bug in calc dot py")

    assert ".env" not in context["files"] and "notes.md" in context["files"]
    by_path = {item["path"]: item for item in context["file_contents"]}
    assert set(by_path) == {"calc.py", "test_calc.py"}
    assert by_path["calc.py"]["read_only"] is False
    assert by_path["calc.py"]["content"] == BUGGY
    assert len(by_path["calc.py"]["sha256"]) == 64
    assert by_path["test_calc.py"]["read_only"] is True


def test_planning_context_skips_oversized_file_text(tmp_path):
    workspace = _project(tmp_path)
    (workspace / "big.py").write_bytes(("x = 1\n" * 3000).encode("utf-8"))
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))

    [entry] = planning_file_context(kernel.workspace_manager, "refactor big.py")["file_contents"]
    assert entry == {"path": "big.py", "read_only": False, "too_large_to_include": True}


def test_brain_prompt_carries_current_file_text_and_edit_rules(tmp_path):
    workspace = _project(tmp_path)
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    for tool in (ListWorkspaceFilesTool, ReadWorkspaceFileTool, WriteWorkspaceFileTool):
        kernel.register_tool(tool(kernel.workspace_manager))
    brain = AgentBrain(kernel, enabled=True, provider=object())
    intent = IntentResult(
        intent=IntentType.TASK, confidence=0.9, normalized_text=GOAL,
        entities={"goal": GOAL, "capability_scope": PROJECT_SCOPE}, requires_tools=True,
    )

    payload = json.loads(brain._build_prompt(GOAL, intent))

    contents = {item["path"]: item for item in payload["workspace"]["file_contents"]}
    assert contents["calc.py"]["content"] == BUGGY
    assert "expected_sha256_from_step" in payload["existing_file_rules"]
    assert payload["project_update_example"][1]["expected_sha256_from_step"] == "step-1"


def _task_intent(goal=GOAL):
    return IntentResult(
        intent=IntentType.TASK, confidence=0.9, normalized_text=goal,
        entities={"goal": goal, "capability_scope": PROJECT_SCOPE},
        requires_tools=True, requires_reasoning=True,
    )


def _planner(tmp_path, *plans):
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    for tool in (ListWorkspaceFilesTool, ReadWorkspaceFileTool, WriteWorkspaceFileTool):
        kernel.register_tool(tool(kernel.workspace_manager))
    kernel.register_tool(RunCommandTool(kernel.workspace_manager, runtime=LocalPythonSandbox()))
    brain = SequencePlanBrain(*plans)
    return Planner(kernel.tool_registry, agent_brain=brain), brain


def test_update_must_name_an_earlier_read_of_the_same_file(tmp_path):
    bad = _edit_plan(FIXED)
    bad[1]["expected_sha256_from_step"] = "step-3"  # a later step
    planner, brain = _planner(tmp_path, bad, _edit_plan(FIXED))

    plan = planner.plan(GOAL, intent=_task_intent())

    assert "Project file updates must" in brain.intents[1]["planning_feedback"]
    assert plan.steps[1].metadata["expected_sha256_from_step"] == "step-1"


def test_update_with_no_valid_read_is_rejected(tmp_path):
    bad = _edit_plan(FIXED)
    bad[1]["expected_sha256_from_step"] = "step-9"
    planner, _ = _planner(tmp_path, bad, bad)

    with pytest.raises(PlanningError, match="Project file updates must"):
        planner.plan(GOAL, intent=_task_intent())


def test_empty_expected_stdout_is_kept_as_a_real_condition(tmp_path):
    plan_steps = _edit_plan(FIXED)
    plan_steps[-1]["expected_stdout"] = ""
    planner, _ = _planner(tmp_path, plan_steps)

    plan = planner.plan(GOAL, intent=_task_intent())

    assert plan.steps[-1].metadata["expected_stdout"] == ""


def test_step_references_shift_past_the_users_leading_app_steps():
    commands = [{"tool": "filesystem.write_file", "expected_sha256_from_step": "step-1"}, {"tool": "x"}]
    shifted = Planner._offset_step_references(commands, 3)
    assert shifted[0]["expected_sha256_from_step"] == "step-4"
    assert commands[0]["expected_sha256_from_step"] == "step-1"


def test_repair_plan_does_not_replay_the_users_music_part(tmp_path):
    planner, brain = _planner(tmp_path, _edit_plan(FIXED))
    project_goal = "fix calc.py so the tests pass"
    task = SimpleNamespace(
        goal="open Spotify and play Believer, then " + project_goal,
        constraints=(),
        evidence=[],
        metadata={
            "intent_type": "task",
            "intent_entities": {
                "goal": project_goal,
                "capability_scope": PROJECT_SCOPE,
                "prefix_commands": [{"action": "open", "target": "spotify"}],
            },
        },
    )
    diagnosis = SimpleNamespace(step_id="step-4", category=SimpleNamespace(value="tool_failure"))

    plan = planner.replan(task, diagnosis, "rebuild_plan")

    assert brain.goals == [project_goal]
    assert [step.metadata["tool"] for step in plan.steps] == [
        "filesystem.read_file", "filesystem.write_file", "filesystem.read_file", "system.run_command",
    ]


def test_hud_can_approve_a_file_edit_and_shows_counts_not_code(tmp_path):
    from hud.hud_module import HUDModule
    from tests.test_hud_module import RecordingHUDTransport

    workspace = _project(tmp_path)
    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.workspace_manager.update_project(name="calc", path=str(workspace))
    hud = HUDModule(kernel)
    hud.transport = RecordingHUDTransport()
    request = ToolRequest(
        tool="filesystem.write_file",
        arguments={"path": "test_calc.py", "content": TESTS.replace("== 5", "== -1"), "expected_sha256": "0" * 64},
        request_id="edit-test",
        metadata={"confirm_overwrite": "test_calc.py", "confirm_kind": "test_edit"},
    )
    kernel.approval_manager.request_approval(request, "confirm")
    sent = []
    kernel.event_bus.subscribe("tool_confirmation_response", lambda **event: sent.append(event))
    try:
        hud._publish_state()
        [approval] = hud.state.approvals
        assert approval["can_approve"] is True
        preview = json.loads(approval["preview"])
        assert preview == {"action": "edit test file", "path": "test_calc.py", "lines_added": 1, "lines_removed": 1}
        assert "assert" not in approval["preview"]
        hud.on_transport_message({"type": "hud.approval", "request_id": "edit-test", "approved": True})
        assert sent == [{"request_id": "edit-test", "approved": True}]
    finally:
        kernel.shutdown()


def test_hud_still_cannot_approve_a_plain_write():
    from hud.hud_module import HUDModule
    from tests.test_hud_module import RecordingHUDTransport

    kernel = Kernel()
    hud = HUDModule(kernel)
    hud.transport = RecordingHUDTransport()
    request = ToolRequest(tool="filesystem.write_file", arguments={"path": "a.py", "content": "x"}, request_id="w")
    kernel.approval_manager.request_approval(request, "confirm")
    try:
        hud._publish_state()
        assert hud.state.approvals[0]["can_approve"] is False
    finally:
        kernel.shutdown()
