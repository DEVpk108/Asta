from core.module import Module
from core.intent_router import IntentType

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
    # User message routing
    # ---------------------------------------------------------

    def on_user_message(self, text):

        if not text:
            return

        print(
            f"[AI] User: {text}",
            flush=True,
        )

        intent = self.kernel.intent_router.route(
            text
        )

        print(
            f"[AI] Intent: {intent.value}",
            flush=True,
        )

        # -----------------------------------------------------
        # Direct command
        # -----------------------------------------------------

        if intent == IntentType.COMMAND:

            self.event_bus.emit(
                "command_request",
                text=text,
            )

            return

        # -----------------------------------------------------
        # Memory request
        # -----------------------------------------------------

        if intent == IntentType.MEMORY:

            self.event_bus.emit(
                "memory_request",
                text=text,
            )

            return

        # -----------------------------------------------------
        # Conversation / unknown
        #
        # These still go to the LLM for now.
        # -----------------------------------------------------

        def on_sentence(sentence):

            if not sentence:
                return

            print(
                f"[AI] Sentence: {sentence}",
                flush=True,
            )

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

        # -----------------------------------------------------
        # Full response
        # -----------------------------------------------------

        self.event_bus.emit(
            "assistant_response",
            text=response,
        )