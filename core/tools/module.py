from core.module import Module

from core.contracts import ToolRequest


class ToolRuntimeModule(Module):
    """EventBus adapter for the kernel's tool dispatcher.

    The kernel owns the registry, dispatcher, and authority policy.
    This module only translates tool_request events into tool_result events.
    """

    def __init__(self, kernel):
        super().__init__(
            name="ToolRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

    def initialize(self):
        self.event_bus.subscribe(
            "tool_request",
            self.on_tool_request,
        )

        print("[Tools] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe(
            "tool_request",
            self.on_tool_request,
        )

        print("[Tools] Stopped", flush=True)

    def on_tool_request(self, request):
        if not isinstance(request, ToolRequest):
            print(
                "[Tools] Ignoring invalid tool request.",
                flush=True,
            )
            return

        result = self.kernel.tool_dispatcher.dispatch(request)

        self.event_bus.emit(
            "tool_result",
            result=result,
        )
