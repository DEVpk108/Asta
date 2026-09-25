from types import SimpleNamespace

from core.autonomy import (
    DiagnosisCategory,
    FailureDiagnosis,
    ReplanEngine,
    ReplanStrategy,
)
from core.contracts import Plan, PlanStep, PlanStepStatus


def _diagnosis(category="state_mismatch", requires_user=False):
    return FailureDiagnosis(
        category=DiagnosisCategory(category),
        summary="test diagnosis",
        failed_tool="media.control",
        task_id="task-1",
        step_id="step-1",
        confidence=0.9,
        recommended_action="replan",
        requires_user=requires_user,
    )


def test_replan_engine_waits_for_user_before_provider():
    class Provider:
        def __init__(self):
            self.called = False

        def select_replan_strategy(self, diagnosis, *, plan_step=None):
            self.called = True
            raise AssertionError("provider must not override user boundary")

    provider = Provider()
    engine = ReplanEngine(provider)
    decision = engine.choose(
        _diagnosis("authentication", requires_user=True),
    )

    assert decision.strategy is ReplanStrategy.WAIT_FOR_USER
    assert decision.source == "guard"
    assert provider.called is False


def test_replan_engine_falls_back_for_low_confidence_provider():
    class Provider:
        def select_replan_strategy(self, diagnosis, *, plan_step=None):
            from core.autonomy import ReplanDecision

            return ReplanDecision(
                strategy=ReplanStrategy.FAIL,
                reason="uncertain",
                confidence=0.2,
                source="laya",
            )

    decision = ReplanEngine(Provider()).choose(_diagnosis())

    assert decision.strategy is ReplanStrategy.RESTORE_STATE
    assert decision.source == "fallback"
    assert decision.metadata["low_confidence_provider"] is True


def test_restore_state_replan_builds_restore_and_retry_steps():
    from core.planner import Planner

    class SelectorRegistry:
        def definitions(self):
            return ()

    class MediaManager:
        def application_for_provider(self, provider):
            assert provider == "spotify"
            return "Spotify"

    planner = Planner(
        SelectorRegistry(),
        media_manager=MediaManager(),
    )

    class Selector:
        def select(self, intent):
            return SimpleNamespace(
                name="system.open_application",
            )

    planner.selector = Selector()

    failed = PlanStep(
        id="step-1",
        description="play hanuman chalisa",
        status=PlanStepStatus.FAILED,
        required_capabilities=["media.control"],
        completion_conditions=["media.control reports success"],
        metadata={
            "action": "media",
            "operation": "play",
            "query": "hanuman chalisa",
            "provider": "spotify",
            "tool": "media.control",
        },
    )
    task = SimpleNamespace(
        id="task-1",
        goal="Play Hanuman Chalisa on Spotify",
        constraints=[],
        metadata={"confidence": 0.98},
        plan=Plan(
            goal="Play Hanuman Chalisa on Spotify",
            steps=[failed],
        ),
    )

    plan = planner.replan(
        task,
        _diagnosis(),
        ReplanStrategy.RESTORE_STATE,
    )

    assert [step.id for step in plan.steps] == [
        "replan-restore",
        "replan-retry",
    ]
    assert plan.steps[0].description == "open Spotify"
    assert plan.steps[1].depends_on == ["replan-restore"]
    assert plan.steps[1].metadata["replanned"] is True
