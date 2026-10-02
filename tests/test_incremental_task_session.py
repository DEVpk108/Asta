from core import Kernel
from core.contracts import ToolDefinition, ToolRequest, ToolResult, TaskStatus
from core.task_runtime import TaskRuntimeModule
from core.tools import Tool
from ai.ai_module import AIModule


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
        raise AssertionError("LLM should not be called for deterministic commands")


def test_incremental_session_queues_continuation_until_active_segment_completes():
    kernel = Kernel()
    kernel.register_tool(FakeCapabilityTool())

    tasks = TaskRuntimeModule(kernel)
    ai = AIModule(kernel)
    ai.engine = RecordingEngine()

    tasks.initialize()
    ai.initialize()

    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    session_id = "inc-test-session"

    try:
        kernel.event_bus.emit(
            "incremental_voice_session_started",
            session_id=session_id,
        )

        kernel.event_bus.emit(
            "incremental_user_message",
            text="open calculator",
            commit_id="inc-1",
            session_id=session_id,
        )

        assert len(kernel.task_manager.list()) == 1
        first_task = kernel.task_manager.current()
        assert first_task is not None
        assert first_task.status is TaskStatus.ACTIVE
        assert first_task.metadata["incremental_session"]["session_id"] == session_id
        assert first_task.metadata["incremental_session"]["segment_index"] == 1
        assert len(requests) == 1
        first_request = requests[0]
        first_task_id = first_request.metadata["task_id"]
        assert first_task_id == first_task.id

        kernel.event_bus.emit(
            "incremental_user_message",
            text="close calculator",
            commit_id="inc-2",
            session_id=session_id,
        )

        # The second spoken segment belongs to the same voice session but must
        # not interrupt the first task while it is still running.
        assert len(kernel.task_manager.list()) == 1
        assert len(requests) == 1
        assert first_task.status is TaskStatus.ACTIVE

        first_result_metadata = dict(first_request.metadata)
        kernel.event_bus.emit(
            "tool_result",
            result=ToolResult(
                success=True,
                tool=first_request.tool,
                output={"target": "calculator"},
                metadata=first_result_metadata,
            ),
        )

        assert first_task.status is TaskStatus.COMPLETED
        assert len(kernel.task_manager.list()) == 2
        second_task = kernel.task_manager.current()
        assert second_task is not None
        assert second_task.id != first_task.id
        assert second_task.status is TaskStatus.ACTIVE
        assert second_task.metadata["incremental_session"]["session_id"] == session_id
        assert second_task.metadata["incremental_session"]["segment_index"] == 2
        assert len(requests) == 2
        assert requests[1].metadata["task_id"] == second_task.id

        kernel.event_bus.emit(
            "incremental_voice_session_finished",
            session_id=session_id,
        )

        second_request = requests[1]
        kernel.event_bus.emit(
            "tool_result",
            result=ToolResult(
                success=True,
                tool=second_request.tool,
                output={"target": "calculator"},
                metadata=dict(second_request.metadata),
            ),
        )

        assert second_task.status is TaskStatus.COMPLETED
        assert len(kernel.task_manager.list()) == 2
    finally:
        ai.shutdown()
        tasks.shutdown()


def test_incremental_failed_segment_does_not_run_queued_continuation():
    kernel = Kernel()
    kernel.register_tool(FakeCapabilityTool())

    tasks = TaskRuntimeModule(kernel)
    ai = AIModule(kernel)
    ai.engine = RecordingEngine()

    tasks.initialize()
    ai.initialize()

    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    session_id = "inc-failure-session"

    try:
        kernel.event_bus.emit(
            "incremental_voice_session_started",
            session_id=session_id,
        )
        kernel.event_bus.emit(
            "incremental_user_message",
            text="open calculator",
            commit_id="inc-1",
            session_id=session_id,
        )
        first_task = kernel.task_manager.current()
        assert first_task is not None
        assert len(requests) == 1

        kernel.event_bus.emit(
            "incremental_user_message",
            text="close calculator",
            commit_id="inc-2",
            session_id=session_id,
        )
        assert len(requests) == 1

        # RecoveryManager may retry/replan a failed step before the task
        # reaches a terminal FAILED state. Keep feeding failures through every
        # recovery-generated request, while ensuring the queued continuation
        # never starts as a new task.
        failure_count = 0
        while first_task.status is TaskStatus.ACTIVE and failure_count < 8:
            request = requests[-1]
            kernel.event_bus.emit(
                "tool_result",
                result=ToolResult(
                    success=False,
                    tool=request.tool,
                    error="simulated failure",
                    metadata=dict(request.metadata),
                ),
            )
            failure_count += 1

        assert first_task.status is TaskStatus.FAILED
        assert len(kernel.task_manager.list()) == 1
    finally:
        ai.shutdown()
        tasks.shutdown()
