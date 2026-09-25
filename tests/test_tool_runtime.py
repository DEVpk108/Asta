from core import Kernel
from core.contracts import ToolRequest
from core.tools import EchoTool, ToolRuntimeModule


class ResultCollector:
    def __init__(self):
        self.results = []

    def on_result(self, result):
        self.results.append(result)


def test_tool_runtime_module_dispatches_event():
    kernel = Kernel()
    kernel.register_tool(EchoTool())

    tools = ToolRuntimeModule(kernel)
    collector = ResultCollector()

    kernel.register_module(tools)
    kernel.event_bus.subscribe(
        "tool_result",
        collector.on_result,
    )

    kernel.start()

    try:
        request = ToolRequest(
            tool="echo",
            arguments={"text": "hello from event bus"},
            request_id="runtime-1",
        )

        kernel.event_bus.emit(
            "tool_request",
            request=request,
        )

        assert len(collector.results) == 1
        assert collector.results[0].success is True
        assert collector.results[0].output == "hello from event bus"
        assert collector.results[0].metadata["request_id"] == "runtime-1"
    finally:
        kernel.shutdown()


def test_tool_runtime_does_not_advance_compound_sequences():
    from core.contracts import ToolDefinition, ToolResult
    from core.tools import Tool

    class FakeSequenceTool(Tool):
        @property
        def definition(self):
            return ToolDefinition(
                name="test.sequence",
                description="Sequence test capability.",
                input_schema={"type": "object", "properties": {}},
                risk_level="low",
                requires_confirmation=False,
                metadata={"actions": ["open"]},
            )

        def execute(self, request):
            return ToolResult(
                success=True,
                tool=self.definition.name,
                output={"ok": True},
            )

    kernel = Kernel()
    kernel.register_tool(FakeSequenceTool())
    runtime = ToolRuntimeModule(kernel)
    runtime.initialize()

    observed_requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: observed_requests.append(request),
    )

    try:
        request = ToolRequest(
            tool="test.sequence",
            arguments={"target": "x"},
            request_id="sequence-runtime-1",
            metadata={
                "sequence": [
                    {"action": "open", "target": "x"},
                    {"action": "open", "target": "y"},
                ],
                "sequence_index": 0,
            },
        )
        kernel.event_bus.emit("tool_request", request=request)

        assert len(observed_requests) == 1
        assert observed_requests[0] is request
        assert observed_requests[0].request_id == "sequence-runtime-1"
    finally:
        runtime.shutdown()
