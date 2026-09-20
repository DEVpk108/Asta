from core import Kernel
from core.contracts import ToolDefinition, ToolRequest, ToolResult, TaskStatus
from core.module import Module
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
        assert task.plan.status.value == "ready"
        assert len(task.plan.steps) == 1
        assert task.plan.steps[0].description == "open calculator"
        assert task.plan.steps[0].metadata["tool"] == "test.capability"
    finally:
        _shutdown(tasks, ai, tools)
