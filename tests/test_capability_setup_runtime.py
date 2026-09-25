from core import Kernel
from core.contracts import Plan, PlanStep, PlanStepStatus, TaskStatus, ToolResult
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
        status="active",
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
