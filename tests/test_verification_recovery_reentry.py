"""A failed postcondition belongs to one real execution, not another bus result."""
from types import SimpleNamespace

import pytest

from core import Kernel
from core.autonomy import RecoveryManager
from core.autonomy.verification import VerificationResult, VerificationStatus
from core.autonomy.replanning import ReplanDecision, ReplanStrategy
from core.contracts import Plan, PlanStep, TaskStatus, ToolResult
from core.task_runtime import TaskRuntimeModule
from core.tools import ToolRuntimeModule
from tests.test_task_runtime import FakeCapabilityTool


@pytest.fixture
def runtime():
    kernel = Kernel()
    kernel.register_tool(FakeCapabilityTool())
    tasks = TaskRuntimeModule(kernel)
    tools = ToolRuntimeModule(kernel)
    tasks.initialize()
    tools.initialize()
    step = PlanStep(id="step-1", description="open calculator", metadata={
        "tool": "test.capability", "action": "open", "target": "calculator",
        "verification": "test.postcondition",
    })
    task = kernel.task_manager.create("open calculator", plan=Plan(
        goal="open calculator", steps=[step]), pending_steps=[step.description])
    kernel.replan_engine = SimpleNamespace(choose=lambda *args, **kwargs:
        ReplanDecision(strategy=ReplanStrategy.FAIL, reason="inert regression fixture"))
    events = []
    kernel.event_bus.subscribe("tool_result", lambda result: events.append(result))
    failures = []
    kernel.event_bus.subscribe("task_verification_failed", lambda **data: failures.append(data))
    decisions = []
    decide = kernel.recovery_manager.decide

    def record_decision(task, result, *, step_id=None):
        decision = decide(task, result, step_id=step_id)
        decisions.append((decision, result, len(task.evidence)))
        return decision

    kernel.recovery_manager.decide = record_decision
    yield kernel, tasks, task, step, events, failures, decisions
    tools.shutdown()
    tasks.shutdown()
    kernel.shutdown()


def _verify_sequence(kernel, statuses):
    remaining = iter(statuses)

    def verify(task, step, result):
        return VerificationResult(status=next(remaining), summary="observed outcome",
            source="test_observer", task_id=task.id, step_id=step.id)

    kernel.verification_engine = SimpleNamespace(verify=verify)


def _start(kernel, tasks, task, step):
    kernel.event_bus.emit("tool_request", request=tasks.build_plan_request(task, step))


def test_one_failed_verification_uses_one_retry_and_one_evidence_entry(runtime):
    kernel, tasks, task, step, events, failures, decisions = runtime
    _verify_sequence(kernel, [VerificationStatus.FAILED, VerificationStatus.VERIFIED])
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.COMPLETED
    assert [item[0].attempt for item in decisions] == [1]
    assert decisions[0][2] == 0  # history excludes the current attempt
    assert len(events) == 2 and all(result.success for result in events)
    assert all(not result.metadata.get('verification_failure') for result in events)
    assert len(failures) == 1
    assert len(task.evidence) == 2
    assert [item['verification']['status'] for item in task.evidence] == ['failed', 'verified']
    first = task.evidence[0]
    assert first['success'] is True  # tool success remains distinct from outcome
    assert first['output'] == {'target': 'calculator'}
    assert first['request_id'] and first['recovery']['attempt'] == 1
    assert decisions[0][1].output == first['output']
    assert decisions[0][1].metadata['request_id'] == first['request_id']
    assert decisions[0][1].metadata['verification_failure'] is True


def test_repeated_verification_failure_exhausts_budget_without_double_count(runtime):
    kernel, tasks, task, step, events, failures, decisions = runtime
    _verify_sequence(kernel, [VerificationStatus.FAILED, VerificationStatus.FAILED])
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.FAILED
    assert [item[0].attempt for item in decisions] == [1, 2]
    assert [item[2] for item in decisions] == [0, 1]
    assert len(events) == 2 and all(result.success for result in events)
    assert len(failures) == 2 and len(task.evidence) == 2
    assert [item['recovery']['action'] for item in task.evidence] == ['retry', 'replan']
    assert all(item['verification']['status'] == 'failed' for item in task.evidence)


def test_unknown_verification_pauses_without_entering_recovery(runtime):
    kernel, tasks, task, step, events, failures, decisions = runtime
    _verify_sequence(kernel, [VerificationStatus.UNKNOWN])
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.PAUSED
    assert len(events) == 1 and len(task.evidence) == 1
    assert failures == [] and decisions == []


def test_failed_verification_without_diagnosis_still_records_current_attempt(runtime):
    kernel, tasks, task, step, events, failures, decisions = runtime
    kernel.recovery_manager = RecoveryManager(max_retries=0)
    kernel.diagnosis_engine = None
    _verify_sequence(kernel, [VerificationStatus.FAILED])
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.PAUSED
    assert len(events) == 1 and len(task.evidence) == 1
    assert task.evidence[0]['recovery']['attempt'] == 1


def test_ordinary_tool_failure_is_recorded_once_and_keeps_retry_budget(runtime, monkeypatch):
    kernel, tasks, task, step, events, failures, decisions = runtime
    step.metadata.pop('verification')
    tool = kernel.tool_registry.get('test.capability')
    execute = tool.execute
    calls = []

    def fail_once(request):
        calls.append(request)
        if len(calls) == 1:
            return ToolResult(success=False, tool=request.tool, error='temporary failure')
        return execute(request)

    monkeypatch.setattr(tool, 'execute', fail_once)
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.COMPLETED
    assert len(events) == 2 and len(task.evidence) == 2
    assert [item[0].attempt for item in decisions] == [1]
    assert failures == []


def test_terminal_recovery_records_ordinary_tool_failure(runtime, monkeypatch):
    from core.autonomy import RecoveryAction, RecoveryDecision
    kernel, tasks, task, step, events, failures, decisions = runtime
    monkeypatch.setattr(kernel.tool_registry.get('test.capability'), 'execute',
        lambda request: ToolResult(success=False, tool=request.tool, error='unrecoverable'))
    kernel.recovery_manager = SimpleNamespace(decide=lambda *args, **kwargs:
        RecoveryDecision(action=RecoveryAction.FAIL, reason='fixture', failed_tool='test.capability'))
    _start(kernel, tasks, task, step)
    assert task.status is TaskStatus.FAILED
    assert len(task.evidence) == 1
    assert task.evidence[0]['error'] == 'unrecoverable'


@pytest.mark.parametrize("action,missing_engine,expected_status", [
    ("wait_for_user", None, TaskStatus.PAUSED),
    ("fail", None, TaskStatus.FAILED),
    ("replan", "diagnosis_engine", TaskStatus.PAUSED),
    ("replan", "replan_engine", TaskStatus.PAUSED),
])
def test_verification_recovery_exit_records_exactly_one_attempt(
    runtime, action, missing_engine, expected_status,
):
    from core.autonomy import RecoveryAction, RecoveryDecision
    kernel, tasks, task, step, events, failures, decisions = runtime
    if missing_engine:
        setattr(kernel, missing_engine, None)
    kernel.recovery_manager = SimpleNamespace(decide=lambda *args, **kwargs:
        RecoveryDecision(action=RecoveryAction(action), reason='fixture',
                         failed_tool='test.capability'))
    _verify_sequence(kernel, [VerificationStatus.FAILED])
    _start(kernel, tasks, task, step)
    assert task.status is expected_status
    assert len(events) == len(failures) == len(task.evidence) == 1
    assert events[0].success is True
    assert task.evidence[0]['verification']['status'] == 'failed'
    assert task.evidence[0]['recovery']['action'] == action


@pytest.mark.parametrize("tool", ["system.run_command", "project.open_program"])
def test_deterministic_execution_verification_diagnoses_without_blind_retry(runtime, tool):
    kernel, tasks, task, step, events, failures, decisions = runtime
    step.metadata['tool'] = tool
    _verify_sequence(kernel, [VerificationStatus.FAILED])
    requests = []
    kernel.event_bus.subscribe("tool_request", lambda request: requests.append(request))
    kernel.event_bus.emit("tool_result", result=ToolResult(
        success=True, tool=tool, output={"stdout": "wrong output"}, duration_seconds=0.25,
        metadata={"task_id": task.id, "plan_step_id": step.id,
                  "task_step": step.description, "request_id": "original-execution"},
    ))
    assert task.status is TaskStatus.FAILED
    assert requests == []
    assert len(events) == len(failures) == len(task.evidence) == 1
    decision, failure, previous_evidence_count = decisions[0]
    assert decision.action.value == 'replan' and decision.attempt == 1
    assert previous_evidence_count == 0
    assert failure.duration_seconds == 0.25
    assert failure.output == {"stdout": "wrong output"}
    assert failure.metadata["request_id"] == "original-execution"
