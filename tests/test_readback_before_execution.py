"""Every project write must be observed before code execution or overwrite."""
import copy

import pytest

from core.contracts import PlanStep
from core.planner import Planner, PlanningError


def _write(path="a.py", content="print(1)\n", **extra):
    return {"tool": "filesystem.write_file", "action": "write_file",
            "path": path, "content": content, **extra}


def _read(path="a.py", content="print(1)\n"):
    return {"tool": "filesystem.read_file", "action": "read_file", "path": path,
            "verification": "filesystem.file_content", "expected_path": path,
            "expected_content": content}


def _execute(tool="system.run_command", **extra):
    if tool == "project.open_program":
        return {"tool": tool, "action": "open_program", "path": "a.py", **extra}
    return {"tool": tool, "action": "run", "target": "python",
            "arguments": ["a.py"], "verification": "command.output",
            "expected_stdout": "1", **extra}


def _steps(commands):
    return [PlanStep(id=f"step-{index}", description=command["tool"], metadata=command)
            for index, command in enumerate(commands, 1)]


@pytest.mark.parametrize("tool", ["system.run_command", "project.open_program"])
@pytest.mark.parametrize("named", [True, False])
def test_normalizer_inserts_readback_before_execution_even_with_late_read(tool, named):
    commands = [_write(), _execute(tool), _read()]
    if not named:
        for command in commands:
            command.pop("tool")
    result = Planner._complete_project_commands(commands)
    assert len(result) == 4
    assert result[1]["tool"] == "filesystem.read_file"
    assert result[1]["verification"] == "filesystem.file_content"
    assert result[1]["expected_content"] == commands[0]["content"]
    assert result[1]["path"] == result[1]["expected_path"] == "a.py"
    assert result[2].get("tool") == tool or result[2]["action"] == "run"


@pytest.mark.parametrize("tool", ["system.run_command", "project.open_program"])
def test_raw_validator_rejects_late_readback_without_relying_on_normalizer(tool):
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps([_write(), _execute(tool), _read()]))


@pytest.mark.parametrize("tool", ["system.run_command", "project.open_program"])
def test_validator_accepts_corrected_plan(tool):
    commands = Planner._complete_project_commands([_write(), _execute(tool), _read()])
    Planner._validate_project_task_verification(_steps(commands))


def test_existing_safe_readback_is_reused_without_adding_another_one():
    commands = [_write(), {"tool": "filesystem.list_files", "path": "."}, _read(), _execute()]
    result = Planner._complete_project_commands(commands)
    assert len(result) == len(commands)
    Planner._validate_project_task_verification(_steps(result))


def test_no_read_capability_does_not_fabricate_a_safe_plan():
    commands = Planner._complete_project_commands(
        [_write(), _execute(), _read()], can_insert_readback=False)
    assert len(commands) == 3
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps(commands))


def test_all_written_files_are_verified_before_first_execution():
    commands = [_write("a.py", "A"), _write("b.py", "B"), _execute(),
                _read("a.py", "A"), _read("b.py", "B")]
    result = Planner._complete_project_commands(commands)
    assert [step["tool"] for step in result[:5]] == [
        "filesystem.write_file", "filesystem.read_file",
        "filesystem.write_file", "filesystem.read_file", "system.run_command",
    ]
    Planner._validate_project_task_verification(_steps(result))


def test_verifying_one_file_does_not_allow_running_with_another_unverified_write():
    commands = [_write("a.py", "A"), _read("a.py", "A"), _write("b.py", "B"),
                _execute(), _read("b.py", "B")]
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps(commands))


def test_execution_path_is_irrelevant_to_the_readback_barrier():
    commands = [_write("config.json", "{}"), _execute(path="unrelated.py"),
                _read("config.json", "{}")]
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps(commands))


def test_second_write_requires_a_new_readback_even_when_contents_are_identical():
    commands = [_write(), _read(), _write(), _execute(), _read()]
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps(commands))
    Planner._validate_project_task_verification(
        _steps(Planner._complete_project_commands(commands)))


def test_a_write_cannot_borrow_a_readback_after_the_file_is_overwritten():
    with pytest.raises(PlanningError, match="Project file writes must.*before"):
        Planner._validate_project_task_verification(_steps([_write(), _write(), _read()]))


def test_inserted_readbacks_remap_hash_references_and_do_not_modify_input():
    commands = [_write("a.py", "A"), _execute(), _read("a.py", "A"),
                _write("a.py", "B", expected_sha256_from_step="step-3"),
                _read("a.py", "B"), _execute()]
    original = copy.deepcopy(commands)
    result = Planner._complete_project_commands(commands)
    assert commands == original
    assert result[4]["expected_sha256_from_step"] == "step-4"
    assert result[3]["tool"] == "filesystem.read_file"
    Planner._validate_project_task_verification(_steps(result))
    assert Planner._complete_project_commands(result) == result


def test_two_insertions_remap_later_hash_reference_twice():
    commands = [_write("a.py", "A"), _write("b.py", "B"), _execute(),
                _read("a.py", "A"), _write("a.py", "C", expected_sha256_from_step="step-4"),
                _read("a.py", "C"), _read("b.py", "B")]
    result = Planner._complete_project_commands(commands)
    update = next(step for step in result if step.get("content") == "C")
    assert update["expected_sha256_from_step"] == "step-6"
    Planner._validate_project_task_verification(_steps(result))


def test_readback_matches_windows_separators_before_execution():
    commands = [_write("src\\a.py"), _read("src/a.py"), _execute()]
    result = Planner._complete_project_commands(commands)
    assert len(result) == 3
    Planner._validate_project_task_verification(_steps(result))


def test_correct_order_does_not_excuse_wrong_content():
    with pytest.raises(PlanningError, match="verified readback must match"):
        Planner._validate_project_task_verification(_steps([_write(), _read(content="wrong"), _execute()]))


def test_real_runtime_verifies_before_requesting_execution_approval(tmp_path, monkeypatch):
    from core.contracts import TaskStatus
    from tests.test_workspace_project_task import (
        FILE_CONTENT, GOAL, _build_runtime, _happy_plan, _shutdown,
    )

    workspace = tmp_path / "project"
    workspace.mkdir()
    original = _happy_plan()
    late_plan = [original[0], original[1], original[3], original[2]]
    kernel, tasks, ai, tools = _build_runtime(
        workspace, tmp_path / "authority.json", late_plan, authorize_run=False)
    observed = []
    verifier = kernel.verification_engine.verify
    command_tool = kernel.tool_registry.get("system.run_command")
    execute = command_tool.execute

    def verify(task, step, result):
        check = verifier(task, step, result)
        if step.metadata.get("verification") == "filesystem.file_content":
            observed.append(("readback", check.verified))
        return check

    def run(request):
        assert ("readback", True) in observed
        observed.append(("run", True))
        return execute(request)  # fixture container runtime, no real execution

    monkeypatch.setattr(kernel.verification_engine, "verify", verify)
    monkeypatch.setattr(command_tool, "execute", run)
    try:
        kernel.event_bus.emit("user_message", GOAL)
        assert observed == [("readback", True)]
        pending = kernel.approval_manager.list_pending()
        assert len(pending) == 1 and pending[0].request.tool == "system.run_command"
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
        kernel.event_bus.emit("tool_confirmation_response",
                              request_id=pending[0].request.request_id, approved=True)
        assert observed[:2] == [("readback", True), ("run", True)]
        assert kernel.task_manager.list()[0].status is TaskStatus.COMPLETED
    finally:
        _shutdown(tasks, ai, tools)
        kernel.shutdown()


def test_failed_readback_blocks_execution_in_real_runtime(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from core.autonomy.replanning import ReplanDecision, ReplanStrategy
    from core.contracts import ToolResult
    from tests.test_workspace_project_task import GOAL, _build_runtime, _happy_plan, _shutdown

    workspace = tmp_path / "project"
    workspace.mkdir()
    original = _happy_plan()
    kernel, tasks, ai, tools = _build_runtime(
        workspace, tmp_path / "authority.json",
        [original[0], original[1], original[3], original[2]])
    reads = []
    runs = []
    read_tool = kernel.tool_registry.get("filesystem.read_file")
    read = read_tool.execute

    def corrupted_readback(request):
        actual = read(request)
        reads.append(request.request_id)
        return ToolResult(
            success=True, tool=actual.tool,
            output={**actual.output, "content": "unexpected file contents"},
            metadata=actual.metadata,
        )

    monkeypatch.setattr(read_tool, "execute", corrupted_readback)
    monkeypatch.setattr(kernel.tool_registry.get("system.run_command"), "execute",
                        lambda request: runs.append(request) or
                        ToolResult(success=True, tool=request.tool))
    # Keep this test about the execution barrier, not the separately tracked
    # verification-recovery re-entry bug. Choose terminal failure deterministically.
    kernel.replan_engine = SimpleNamespace(choose=lambda *args, **kwargs:
        ReplanDecision(strategy=ReplanStrategy.FAIL, reason="fixture stops repair"))
    try:
        kernel.event_bus.emit("user_message", GOAL)
        assert reads
        assert runs == []
        assert any(
            item.get("verification", {}).get("status") == "failed"
            for item in kernel.task_manager.list()[0].evidence
        )
    finally:
        _shutdown(tasks, ai, tools)
        kernel.shutdown()
