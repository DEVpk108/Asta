from core.module import Module


class HUDModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="HUD",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

    def initialize(self):
        self.event_bus.subscribe(
            "assistant_response",
            self.on_assistant_response,
        )

    def on_assistant_response(self, text):
        print(
            f"[HUD] {text}",
            flush=True,
        )

    def shutdown(self):
        self.event_bus.unsubscribe(
            "assistant_response",
            self.on_assistant_response,
        )