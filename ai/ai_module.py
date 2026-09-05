from core.module import Module
from core.contracts import IntentType, IntentResult

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
        print("[AI] Initializing...", flush=True)

        self.event_bus.subscribe(
            "user_message",
            self.on_user_message,
        )

        print("[AI] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe(
            "user_message",
            self.on_user_message,
        )

        print("[AI] Stopped", flush=True)

    # ---------------------------------------------------------
    # User message
    # ---------------------------------------------------------

    def on_user_message(self, text):

        if not text:
            return

        print(
            f"[AI] User: {text}",
            flush=True,
        )

        # -----------------------------------------------------
        # Intent analysis
        # -----------------------------------------------------

        result: IntentResult = (
            self.kernel.intent_router.analyze(text)
        )

        print(
            f"[AI] Intent: {result.intent.value} "
            f"(confidence={result.confidence:.2f}, "
            f"classifier={result.classifier})",
            flush=True,
        )

        # -----------------------------------------------------
        # Direct command
        # -----------------------------------------------------

        if result.intent == IntentType.COMMAND:

            self.event_bus.emit(
                "command_request",
                intent=result,
            )

            return

        # -----------------------------------------------------
        # Memory request
        # -----------------------------------------------------

        if result.intent == IntentType.MEMORY:

            self.event_bus.emit(
                "memory_request",
                intent=result,
            )

            return

        # -----------------------------------------------------
        # Conversation / task / unknown
        # -----------------------------------------------------

        def on_sentence(sentence):

            if not sentence:
                return

            self.event_bus.emit(
                "assistant_sentence",
                text=sentence,
            )

        response = self.engine.generate_response(
            text,
            on_sentence=on_sentence,
        )

        if not response:
            print(
                "[AI] No response generated.",
                flush=True,
            )
            return

        self.event_bus.emit(
            "assistant_response",
            text=response,
        )