# core/intent_router.py

from enum import Enum


class IntentType(Enum):
    CONVERSATION = "conversation"
    COMMAND = "command"
    MEMORY = "memory"
    UNKNOWN = "unknown"


class IntentRouter:

    def route(self, text: str) -> IntentType:

        if not text:
            return IntentType.UNKNOWN

        text = text.strip().lower()

        # ---------------------------------------------------------
        # Memory
        # ---------------------------------------------------------

        memory_phrases = (
            "remember that",
            "remember this",
            "don't forget",
            "save this",
            "keep this in mind",
        )

        if text.startswith(memory_phrases):
            return IntentType.MEMORY

        # ---------------------------------------------------------
        # Direct commands
        # ---------------------------------------------------------

        command_phrases = (
            "open ",
            "close ",
            "launch ",
            "start ",
            "run ",
            "stop ",
            "mute",
            "unmute",
            "take a screenshot",
            "screenshot",
        )

        if text.startswith(command_phrases):
            return IntentType.COMMAND

        # ---------------------------------------------------------
        # Normal conversation
        # ---------------------------------------------------------

        conversation_phrases = (
            "who are you",
            "what are you",
            "how are you",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good evening",
            "good night",
            "thank you",
            "thanks",
        )

        if text.startswith(conversation_phrases):
            return IntentType.CONVERSATION

        # ---------------------------------------------------------
        # Default
        #
        # Unknown requests will eventually go through the LLM.
        # ---------------------------------------------------------

        return IntentType.UNKNOWN