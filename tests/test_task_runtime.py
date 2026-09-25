from core import Kernel
from core.contracts import ToolDefinition, ToolRequest, ToolResult, TaskStatus
from core.task_runtime import TaskRuntimeModule
from core.tools import Tool, ToolRuntimeModule
from core.media import MediaManager


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


def test_failed_tool_exhausts_replan_budget_without_looping_forever():
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
        assert task.evidence[-1]["recovery"]["action"] == "replan"
        assert task.evidence[-1]["replan_guard"]["max_replans"] == 3
        assert task.evidence[-1]["replan_guard"]["attempt"] == 4

        kernel.tool_dispatcher.dispatch = original
    finally:
        _shutdown(tasks, ai, tools)


def test_failed_tool_can_complete_after_autonomous_replan():
    kernel, tasks, ai, tools = _build_runtime()
    calls = 0
    try:
        original = kernel.tool_dispatcher.dispatch

        def recover_on_replan(request, *, confirmed=False):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return ToolResult(
                    success=False,
                    tool=request.tool,
                    error="simulated state failure",
                )
            return ToolResult(
                success=True,
                tool=request.tool,
                output={"target": request.arguments.get("target")},
            )

        kernel.tool_dispatcher.dispatch = recover_on_replan
        kernel.event_bus.emit("user_message", "open calculator")

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert task.plan.status.value == "completed"
        assert task.metadata["replan_attempts"] == 1
        assert calls == 3
        replans = [
            item
            for item in task.evidence
            if "replan" in item
        ]
        assert replans
        assert replans[-1]["replan"]["strategy"] == "rebuild_plan"

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
        assert second.metadata["sequence_index"] == 1
        assert second.metadata["planner"] == "task_runtime"
        assert "sequence" not in second.metadata
    finally:
        _shutdown(tasks, ai, tools)


def test_task_runtime_remembers_successful_opened_application():
    from core.applications import ApplicationRecord

    kernel = Kernel()
    tasks = TaskRuntimeModule(kernel)
    record = ApplicationRecord(
        name="Spotify",
        launch_target=r"shell:AppsFolder\Spotify.App",
        provider="windows.start_apps",
        app_id="Spotify.App",
    )

    original_resolve = kernel.application_manager.resolve
    original_remember = kernel.application_manager.remember_opened
    captured = []

    kernel.application_manager.resolve = lambda target: record
    kernel.application_manager.remember_opened = captured.append

    try:
        tasks._remember_opened_application(
            ToolResult(
                success=True,
                tool="system.open_application",
                output={"target": "spotify"},
            )
        )
    finally:
        kernel.application_manager.resolve = original_resolve
        kernel.application_manager.remember_opened = original_remember

    assert captured == [record]



class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open an application for tests.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request):
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={"target": request.arguments["target"]},
        )


class FakeMediaPlayTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.media",
            description="Play media for tests.",
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
            requires_confirmation=False,
            metadata={"actions": ["media"]},
        )

    def execute(self, request):
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={"operation": request.arguments["operation"]},
        )


def test_task_runtime_executes_planner_generated_media_sequence():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    kernel.register_tool(FakeMediaPlayTool())

    # Keep the real media-provider registry so the planner can derive the
    # application preparation step from the provider name.
    kernel.media_manager = MediaManager()
    kernel.planner = __import__(
        "core.planner",
        fromlist=["Planner"],
    ).Planner(
        kernel.tool_registry,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )

    tasks = TaskRuntimeModule(kernel)
    tools = ToolRuntimeModule(kernel)
    ai = __import__("ai.ai_module", fromlist=["AIModule"]).AIModule(kernel)
    ai.engine = RecordingEngine()

    tasks.initialize()
    ai.initialize()
    tools.initialize()

    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    try:
        kernel.event_bus.emit(
            "user_message",
            "play hanuman chalisa on spotify",
        )

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert [step.description for step in task.plan.steps] == [
            "open Spotify",
            "play hanuman chalisa",
        ]
        # EventBus dispatch is synchronous and the first tool result can
        # advance the task before a later observer receives the original event.
        # Assert semantic plan order by step id instead of observer callback order.
        requests_by_step = {
            request.metadata["plan_step_id"]: request
            for request in requests
        }
        assert requests_by_step["step-1"].tool == "test.open"
        assert requests_by_step["step-2"].tool == "test.media"
        assert requests_by_step["step-1"].arguments == {"target": "Spotify"}
        assert requests_by_step["step-2"].arguments == {
            "operation": "play",
            "query": "hanuman chalisa",
            "provider": "spotify",
        }
    finally:
        _shutdown(tasks, ai, tools)
