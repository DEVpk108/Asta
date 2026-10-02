from core.autonomy import VerificationEngine
from core.autonomy.verification import VerificationStatus
from core.contracts import PlanStep, ToolResult


class FakeApplicationManager:
    def __init__(self, running=True):
        self.running = running
        self.queries = []

    def is_application_running(self, query):
        self.queries.append(query)
        return self.running


class FakeTask:
    id = "task-verification-test"


def test_application_running_verification_uses_process_state():
    manager = FakeApplicationManager(running=True)
    engine = VerificationEngine(
        application_manager=manager,
        poll_attempts=1,
        poll_delay=0,
    )
    step = PlanStep(
        id="step-1",
        description="open calculator",
        metadata={
            "target": "calculator",
            "verification": "application.running",
        },
    )
    result = ToolResult(
        success=True,
        tool="system.open_application",
        output={
            "target": "calculator",
            "resolved_target": "shell:AppsFolder\\Calculator.App",
        },
    )

    verification = engine.verify(FakeTask(), step, result)

    assert verification.status is VerificationStatus.VERIFIED
    assert manager.queries == ["calculator"]


def test_application_running_verification_fails_without_running_process():
    manager = FakeApplicationManager(running=False)
    engine = VerificationEngine(
        application_manager=manager,
        poll_attempts=1,
        poll_delay=0,
    )
    step = PlanStep(
        id="step-1",
        description="open calculator",
        metadata={
            "target": "calculator",
            "verification": "application.running",
        },
    )
    result = ToolResult(
        success=True,
        tool="system.open_application",
        output={"target": "calculator"},
    )

    verification = engine.verify(FakeTask(), step, result)

    assert verification.status is VerificationStatus.FAILED
    assert manager.queries == ["calculator"]
