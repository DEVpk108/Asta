from core import Kernel
from core.contracts import ToolResult
from core.tools import OpenApplicationTool, ToolRuntimeModule


class RecordingEngine:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a command intent")


def test_command_intent_reaches_runtime_without_approval_for_everyday_open(monkeypatch):
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

    class FakeOpenTool(OpenApplicationTool):
        @staticmethod
        def _open(target):
            return None

    kernel.tool_registry.unregister("system.open_application")
    kernel.register_tool(FakeOpenTool())

    runtime.initialize()
    ai.initialize()

    ai.on_user_message("open calculator")

    assert confirmations == []
    assert results
    result = results[-1]
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.tool == "system.open_application"
    assert result.output["target"] == "calculator"
    assert result.output["resolved_target"] == "calc.exe"

    ai.shutdown()
    runtime.shutdown()
