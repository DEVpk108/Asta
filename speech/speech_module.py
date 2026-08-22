from core.module import Module
from .polly_engine import PollyEngine


class SpeechModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = PollyEngine()

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def initialize(self):
        print("[Speech] Initializing...")

        self.event_bus.subscribe(
            "assistant_response",
            self.on_assistant_response,
        )

        print("[Speech] Ready")

    def shutdown(self):
        self.event_bus.unsubscribe(
            "assistant_response",
            self.on_assistant_response,
        )

        print("[Speech] Stopped")

    # ---------------------------------------------------------
    # Event handling
    # ---------------------------------------------------------

    def on_assistant_response(self, text):
        if not text:
            return

        print(
            f"[Speech] {text}",
            flush=True,
        )

        try:
            self.engine.speak(text)

        except Exception as exc:
            print(
                f"[Speech] Error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )