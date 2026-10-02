from core.contracts import AgentTask, Plan, PlanStatus, PlanStep, PlanStepStatus
from core.task_runtime import TaskRuntimeModule
from core.tools.registry import ToolRegistry


class FakeEventBus:
    def subscribe(self, *args, **kwargs):
        pass


class FakeManager:
    def create(self, *args, **kwargs):
        raise NotImplementedError


class FakeKernel:
    def __init__(self):
        self.event_bus = FakeEventBus()
        self.tool_registry = ToolRegistry()
        self.task_manager = FakeManager()


def make_runtime():
    return TaskRuntimeModule(FakeKernel())


def make_task(observations):
    return AgentTask(
        id="task-1",
        goal="click the Create App button",
        status="active",
        plan=Plan(
            goal="click the Create App button",
            status=PlanStatus.ACTIVE,
            steps=[],
        ),
        metadata={"agent_state": {"observations": observations}},
    )


def test_grounded_click_uses_latest_matching_locate_result():
    runtime = make_runtime()
    task = make_task(
        [
            {
                "source": "vision.locate",
                "data": {
                    "success": True,
                    "output": {
                        "target": "Create App",
                        "screen_center": {"x": 760, "y": 400},
                        "confidence": 0.92,
                    },
                },
            }
        ]
    )

    assert runtime._ground_action_arguments(
        task,
        action="click",
        target="Create App",
    ) == {"x": 760, "y": 400}


def test_grounded_click_requires_matching_target():
    runtime = make_runtime()
    task = make_task(
        [
            {
                "source": "vision.locate",
                "data": {
                    "success": True,
                    "output": {
                        "target": "Create App",
                        "screen_center": {"x": 760, "y": 400},
                    },
                },
            }
        ]
    )

    assert runtime._ground_action_arguments(
        task,
        action="click",
        target="Submit",
    ) is None


def test_non_coordinate_actions_need_no_grounding():
    runtime = make_runtime()
    task = make_task([])

    assert runtime._ground_action_arguments(
        task,
        action="open",
        target="calculator",
    ) == {}



def test_planned_grounded_transition_skips_extra_llm_call():
    from core.contracts import PlanStep, PlanStepStatus, ToolResult

    runtime = make_runtime()
    task = make_task([])
    task.plan.steps = [
        PlanStep(
            id="step-1",
            description="locate Create App",
            status=PlanStepStatus.RUNNING,
            metadata={
                "action": "locate",
                "target": "Create App",
                "tool": "vision.locate",
            },
        ),
        PlanStep(
            id="step-2",
            description="click Create App",
            status=PlanStepStatus.PENDING,
            depends_on=["step-1"],
            metadata={
                "action": "click",
                "target": "Create App",
                "tool": "computer.click",
            },
        ),
    ]

    result = ToolResult(
        success=True,
        tool="vision.locate",
        output={
            "target": "Create App",
            "screen_center": {"x": 760, "y": 400},
            "confidence": 0.92,
        },
    )

    assert runtime._should_follow_planned_grounded_action(
        task,
        result,
        "step-1",
    ) is True
