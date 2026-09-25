from core import Kernel
from core.contracts import (\n    Plan,\n    PlanStatus,\n    PlanStep,\n    PlanStepStatus,\n    TaskStatus,\n    ToolDefinition,\n    ToolResult,\n)
from core.task_runtime import TaskRuntimeModule


class FakeSetupManager:
    def __init__(self):
        self.starts = []

    def supports(self, capability):
        return capability == "spotify"

    def start(self, task, *, capability, step_id=None):
        self.starts.append((task.id, capability, step_id))
        return True


def test_setup_required_failure_starts_capability_operator_and_pauses_task():
    kernel = Kernel()
    setup = FakeSetupManager()
    kernel.capability_setup_manager = setup
    tasks = TaskRuntimeModule(kernel)
    tasks.initialize()

    plan = Plan(
        goal="play hanuman chalisa on spotify",
        steps=[
            PlanStep(
                id="step-1",
                description="open Spotify",
                status=PlanStepStatus.COMPLETED,
                required_capabilities=["system.open_application"],
            ),
            PlanStep(
                id="step-2",
                description="play hanuman chalisa",
                status=PlanStepStatus.RUNNING,
                depends_on=["step-1"],
                required_capabilities=["media.control"],
                metadata={
                    "action": "media",
                    "operation": "play",
                    "query": "hanuman chalisa",
                    "provider": "spotify",
                    "tool": "media.control",
                },
            ),
        ],
        status=PlanStatus.ACTIVE,
    )
    task = kernel.task_manager.create(
        "play hanuman chalisa on spotify",
        pending_steps=["play hanuman chalisa"],
        plan=plan,
    )

    try:
        tasks.on_tool_result(
            ToolResult(
                success=False,
                tool="media.control",
                error=(
                    "Spotify playback requires one-time setup. "
                    "Set ASTA_SPOTIFY_CLIENT_ID and authorize A.S.T.A."
                ),
                metadata={
                    "task_id": task.id,
                    "task_step": "play hanuman chalisa",
                    "plan_step_id": "step-2",
                },
            )
        )

        refreshed = kernel.task_manager.get(task.id)
        assert refreshed is not None
        assert refreshed.status is TaskStatus.PAUSED
        assert setup.starts == [(task.id, "spotify", "step-2")]
        assert refreshed.metadata["capability_setup"]["status"] == "running"
        assert refreshed.evidence[-1]["capability_setup"]["started"] is True
    finally:
        tasks.shutdown()

class FakeMediaTool:
    @property
    def definition(self):
        return ToolDefinition(
            name="test.media",
            description="Test media capability.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "query": {"type": "string"},
                    "provider": {"type": "string"},
                },
                "required": ["operation"],
            },
            risk_level="low",
            metadata={"actions": ["media"]},
        )


def test_completed_capability_setup_resumes_the_same_task():
    from core.tools import Tool

    class MediaTool(Tool):
        @property
        def definition(self):
            return FakeMediaTool().definition

        def execute(self, request):
            raise AssertionError("setup resume test should only build the next request")

    kernel = Kernel()
    kernel.register_tool(MediaTool())
    tasks = TaskRuntimeModule(kernel)
    tasks.initialize()

    plan = Plan(
        goal="play hanuman chalisa on spotify",
        steps=[
            PlanStep(
                id="step-1",
                description="open Spotify",
                status=PlanStepStatus.COMPLETED,
                required_capabilities=["system.open_application"],
            ),
            PlanStep(
                id="step-2",
                description="play hanuman chalisa",
                status=PlanStepStatus.READY,
                depends_on=["step-1"],
                required_capabilities=["test.media"],
                metadata={
                    "action": "media",
                    "operation": "play",
                    "query": "hanuman chalisa",
                    "provider": "spotify",
                    "tool": "test.media",
                },
            ),
        ],
        status=PlanStatus.ACTIVE,
    )
    task = kernel.task_manager.create(
        "play hanuman chalisa on spotify",
        pending_steps=["play hanuman chalisa"],
        plan=plan,
    )
    kernel.task_manager.pause(task.id)

    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    try:
        tasks.on_capability_setup_completed(
            {
                "task_id": task.id,
                "capability": "spotify",
                "status": "completed",
                "summary": "Spotify setup complete.",
                "step_id": "step-2",
            }
        )

        refreshed = kernel.task_manager.get(task.id)
        assert refreshed is not None
        assert refreshed.status is TaskStatus.ACTIVE
        assert refreshed.plan.get_step("step-2").status is PlanStepStatus.RUNNING
        assert len(requests) == 1
        assert requests[0].tool == "test.media"
        assert requests[0].metadata["task_id"] == task.id
        assert requests[0].metadata["plan_step_id"] == "step-2"
    finally:
        tasks.shutdown()
