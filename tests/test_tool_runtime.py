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
