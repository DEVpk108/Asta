from core.module import Module

from .openai_engine import AIEngine


class AIModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="AIModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = AIEngine()

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def initialize(self):
        print("[AI] Initializing...")

        self.event_bus.subscribe(
            "user_message",
            self.on_user_message,
        )

        print("[AI] Ready")

    def shutdown(self):
        self.event_bus.unsubscribe(
            "user_message",
            self.on_user_message,
        )

        print("[AI] Stopped")

    # ---------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------

    def on_user_message(self, text):
        if not text:
            return

        print(
            f"[AI] User: {text}",
            flush=True,
        )

        response = self.engine.generate_response(
            text
        )

        if not response:
            print(
                "[AI] No response generated.",
                flush=True,
            )
            return

        print(
            f"[AI] Response: {response}",
            flush=True,
        )

        self.event_bus.emit(
            "assistant_response",
            text=response,
        )