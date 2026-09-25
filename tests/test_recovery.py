from types import SimpleNamespace

from core.autonomy import RecoveryAction, RecoveryManager


def _task(*evidence):
    return SimpleNamespace(
        id="task-1",
        evidence=list(evidence),
    )


def _result(*, error="tool failed", metadata=None):
    return SimpleNamespace(
        tool="test.tool",
        error=error,
        metadata=dict(metadata or {}),
    )


def test_first_failure_is_retryable():
    manager = RecoveryManager(max_retries=1)

    decision = manager.decide(
        _task(),
        _result(),
        step_id="step-1",
    )

    assert decision.action is RecoveryAction.RETRY
    assert decision.attempt == 1


def test_retry_budget_exhaustion_requests_replan():
    manager = RecoveryManager(max_retries=1)

    task = _task(
        {
            "type": "tool_result",
            "tool": "test.tool",
            "success": False,
            "plan_step_id": "step-1",
        },
    )

    decision = manager.decide(
        task,
        _result(),
        step_id="step-1",
    )

    assert decision.action is RecoveryAction.REPLAN
    assert decision.attempt == 2
    assert decision.metadata["retry_budget_exhausted"] is True


def test_authentication_failure_waits_for_user():
    manager = RecoveryManager(max_retries=1)

    decision = manager.decide(
        _task(),
        _result(error="Spotify authentication required"),
        step_id="step-1",
    )

    assert decision.action is RecoveryAction.WAIT_FOR_USER
    assert decision.metadata["requires_user"] is True
