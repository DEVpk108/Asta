from core import Kernel
from core.contracts import IntentType, ToolResult
from core.tools import OpenApplicationTool, ToolRuntimeModule


class RecordingEngine:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a command intent")


def test_command_intent_emits_tool_request_and_reaches_runtime(monkeypatch):
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    runtime = ToolRuntimeModule(kernel)

    ai = __import__("ai.ai_module", fromlist=["AIModule"]).AIModule(kernel)
    ai.engine = RecordingEngine()

    confirmations = []
    results = []

    kernel.event_bus.subscribe(
        "tool_confirmation_required",
        lambda request, reason: confirmations.append((request, reason)),
    )
    kernel.event_bus.subscribe(
        "tool_result",
        lambda result: results.append(result),
    )

    # Keep the real OS opener out of this integration test. The assertion is
    # about the event/request/approval boundary, not platform GUI behavior.
    class FakeOpenTool(OpenApplicationTool):
        def _open(self, target):
            return None

    kernel.tool_registry.unregister("system.open_application")
    kernel.register_tool(FakeOpenTool())

    runtime.initialize()
    ai.initialize()

    ai.on_user_message("open calculator")

    assert confirmations
    request, _ = confirmations[0]
    assert request.tool == "system.open_application"
    assert request.arguments == {"target": "calculator"}

    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id=request.request_id,
        approved=True,
    )

    assert results
    result = results[-1]
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.tool == "system.open_application"
    assert result.output["target"] == "calculator"

    ai.shutdown()
    runtime.shutdown()
