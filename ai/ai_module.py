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

        # -----------------------------------------------------
        # Called whenever LM Studio finishes a sentence.
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

        # -----------------------------------------------------
        # Stream the response.
        # -----------------------------------------------------

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
        # Keep the complete response event too.
        #
        # HUD can use this for the full final response.
        # Speech will use assistant_sentence.
        # -----------------------------------------------------

        self.event_bus.emit(
            "assistant_response",
            text=response,
        )