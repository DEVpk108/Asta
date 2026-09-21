from core import Kernel
from core.contracts import ToolDefinition, ToolRequest, ToolResult, TaskStatus
from core.task_runtime import TaskRuntimeModule
from core.tools import Tool, ToolRuntimeModule


class FakeCapabilityTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.capability",
            description="Execute a deterministic test capability.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open", "close"]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={"target": request.arguments["target"]},
        )


class RecordingEngine:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a command intent")


def _build_runtime():
    kernel = Kernel()
    kernel.register_tool(FakeCapabilityTool())

    tasks = TaskRuntimeModule(kernel)
    tools = ToolRuntimeModule(kernel)
    ai = __import__("ai.ai_module", fromlist=["AIModule"]).AIModule(kernel)
    ai.engine = RecordingEngine()

    tasks.initialize()
    ai.initialize()
    tools.initialize()
    return kernel, tasks, ai, tools


def _shutdown(tasks, ai, tools):
    tools.shutdown()
    ai.shutdown()
    tasks.shutdown()


def test_command_creates_and_completes_agent_task():
    kernel, tasks, ai, tools = _build_runtime()
    try:
        kernel.event_bus.emit("user_message", "open calculator")

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert task.goal == "open calculator"
        assert task.completed_steps == ["open calculator"]
        assert task.pending_steps == []
        assert task.current_step is None
        assert task.plan.status.value == "completed"
        assert task.plan.steps[0].status.value == "completed"
        assert task.evidence
        assert task.evidence[-1]["type"] == "tool_result"
        assert task.evidence[-1]["success"] is True
        assert task.result["tool"] == "test.capability"
    finally:
        _shutdown(tasks, ai, tools)


def test_compound_command_keeps_one_task_across_multiple_tools():
    kernel, tasks, ai, tools = _build_runtime()
    try:
        kernel.event_bus.emit(
            "user_message",
            "open calculator and then close calculator",
        )

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert task.completed_steps == ["open calculator", "close calculator"]
        assert task.pending_steps == []
        assert [step.status.value for step in task.plan.steps] == [
            "completed",
            "completed",
        ]
        assert len(task.evidence) == 2
        assert all(item["success"] for item in task.evidence)
    finally:
        _shutdown(tasks, ai, tools)


def test_command_task_is_created_from_a_plan():
    kernel, tasks, ai, tools = _build_runtime()
    try:
        kernel.event_bus.emit("user_message", "open calculator")

        task = kernel.task_manager.list()[0]
        assert task.plan is not None
        assert task.plan.status.value == "completed"
        assert len(task.plan.steps) == 1
        assert task.plan.steps[0].description == "open calculator"
        assert task.plan.steps[0].metadata["tool"] == "test.capability"
    finally:
        _shutdown(tasks, ai, tools)


def test_failed_tool_marks_plan_step_failed():
    kernel, tasks, ai, tools = _build_runtime()
    try:
        original = kernel.tool_dispatcher.dispatch

        def failed_dispatch(request, *, confirmed=False):
            return ToolResult(
                success=False,
                tool=request.tool,
                error="simulated failure",
            )

        kernel.tool_dispatcher.dispatch = failed_dispatch
        kernel.event_bus.emit("user_message", "open calculator")

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.FAILED
        assert task.plan.status.value == "failed"
        assert task.plan.steps[0].status.value == "failed"

        kernel.tool_dispatcher.dispatch = original
    finally:
        _shutdown(tasks, ai, tools)


def test_task_runtime_controls_compound_plan_execution():
    kernel, tasks, ai, tools = _build_runtime()
    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )
    try:
        kernel.event_bus.emit(
            "user_message",
            "open calculator and then close calculator",
        )

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert [step.status.value for step in task.plan.steps] == [
            "completed",
            "completed",
        ]

        assert len(requests) == 2
        requests_by_step = {
            request.metadata["plan_step_id"]: request
            for request in requests
        }
        assert set(requests_by_step) == {"step-1", "step-2"}

        first = requests_by_step["step-1"]
        second = requests_by_step["step-2"]

        assert first.metadata["sequence_index"] == 0
        assert second.metadata["planner"] == "task_runtime"
        assert "sequence" not in second.metadata
    finally:
        _shutdown(tasks, ai, tools)
