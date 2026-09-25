from types import SimpleNamespace

from core.autonomy import (
    DiagnosisCategory,
    DiagnosisEngine,
    FailureDiagnosis,
    RecoveryAction,
    RecoveryManager,
)


def _task(*evidence):
    return SimpleNamespace(
        id="task-1",
        goal="Play Hanuman Chalisa on Spotify",
        evidence=list(evidence),
    )


def _result(error="Spotify authentication required", output=None):
    return SimpleNamespace(
        tool="media.control",
        error=error,
        output=output,
    )


def test_fallback_diagnoses_authentication_as_user_boundary():
    engine = DiagnosisEngine()

    diagnosis = engine.diagnose(
        _task(),
        _result(),
        recovery=SimpleNamespace(
            to_dict=lambda: {
                "action": RecoveryAction.REPLAN.value,
                "attempt": 2,
            }
        ),
        step_id="step-1",
    )

    assert diagnosis.category is DiagnosisCategory.AUTHENTICATION
    assert diagnosis.recommended_action == "wait_for_user"
    assert diagnosis.requires_user is True
    assert diagnosis.source == "fallback"
    assert diagnosis.step_id == "step-1"


def test_provider_diagnosis_is_preserved():
    expected = FailureDiagnosis(
        category=DiagnosisCategory.STATE_MISMATCH,
        summary="Spotify has no active playback device.",
        failed_tool="media.control",
        task_id="task-1",
        step_id="step-1",
        attempt=2,
        confidence=0.91,
        recommended_action="replan",
        source="test-provider",
    )

    class Provider:
        def diagnose_failure(self, state):
            assert state["task_id"] == "task-1"
            assert state["goal"] == "Play Hanuman Chalisa on Spotify"
            assert state["failed_tool"] == "media.control"
            assert state["attempt"] == 2
            return expected

    engine = DiagnosisEngine(Provider())
    diagnosis = engine.diagnose(
        _task(),
        _result(error="no active device"),
        recovery=SimpleNamespace(
            to_dict=lambda: {
                "action": RecoveryAction.REPLAN.value,
                "attempt": 2,
            }
        ),
        step_id="step-1",
    )

    assert diagnosis is expected


def test_provider_failure_falls_back():
    class BrokenProvider:
        def diagnose_failure(self, state):
            raise RuntimeError("provider unavailable")

    engine = DiagnosisEngine(BrokenProvider())
    diagnosis = engine.diagnose(
        _task(),
        _result(error="connection timeout"),
        step_id="step-1",
    )

    assert diagnosis.category is DiagnosisCategory.TRANSIENT
    assert diagnosis.recommended_action == "retry"
    assert "provider unavailable" in diagnosis.metadata["provider_error"]
