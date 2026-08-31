# core/intent_router.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Intent Types
# ============================================================

class IntentType(Enum):
    CONVERSATION = "conversation"
    COMMAND = "command"
    MEMORY = "memory"
    UNKNOWN = "unknown"


# ============================================================
# Intent Result
# ============================================================

@dataclass(frozen=True)
class IntentResult:

    intent: IntentType

    confidence: float

    normalized_text: str

    entities: dict[str, Any] = field(
        default_factory=dict
    )

    classifier: str = "rules"


# ============================================================
# Intent Router
# ============================================================

class IntentRouter:

    def route(self, text: str) -> IntentType:
        """
        Backwards-compatible API.

        Existing ASTA code can continue using:

            router.route(text)

        while the new architecture uses:

            router.analyze(text)
        """

        return self.analyze(text).intent

    # --------------------------------------------------------
    # Main analysis entry point
    # --------------------------------------------------------

    def analyze(self, text: str) -> IntentResult:

        if not text:

            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                normalized_text="",
                classifier="rules",
            )

        normalized = self._normalize(text)

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        memory_phrases = (
            "remember that",
            "remember this",
            "don't forget",
            "do not forget",
            "save this",
            "keep this in mind",
        )

        if normalized.startswith(memory_phrases):

            return IntentResult(
                intent=IntentType.MEMORY,
                confidence=0.98,
                normalized_text=normalized,
                entities=self._extract_memory_entities(
                    normalized
                ),
                classifier="rules",
            )

        # ----------------------------------------------------
        # Direct commands
        # ----------------------------------------------------

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

        if normalized.startswith(command_phrases):

            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities=self._extract_command_entities(
                    normalized
                ),
                classifier="rules",
            )

        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

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

        if normalized.startswith(conversation_phrases):

            return IntentResult(
                intent=IntentType.CONVERSATION,
                confidence=0.98,
                normalized_text=normalized,
                classifier="rules",
            )

        # ----------------------------------------------------
        # Unknown
        #
        # Future:
        #
        # UNKNOWN
        #    ↓
        # Small LLM classifier
        #    ↓
        # IntentResult
        # ----------------------------------------------------

        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.20,
            normalized_text=normalized,
            classifier="rules",
        )

    # ========================================================
    # Normalization
    # ========================================================

    @staticmethod
    def _normalize(text: str) -> str:

        return " ".join(
            text.strip().lower().split()
        )

    # ========================================================
    # Entity Extraction
    # ========================================================

    @staticmethod
    def _extract_command_entities(
        text: str,
    ) -> dict[str, Any]:

        entities: dict[str, Any] = {}

        if text.startswith("open "):

            entities["action"] = "open"
            entities["target"] = text[5:].strip()

        elif text.startswith("launch "):

            entities["action"] = "launch"
            entities["target"] = text[7:].strip()

        elif text.startswith("start "):

            entities["action"] = "start"
            entities["target"] = text[6:].strip()

        elif text.startswith("close "):

            entities["action"] = "close"
            entities["target"] = text[6:].strip()

        elif text.startswith("run "):

            entities["action"] = "run"
            entities["target"] = text[4:].strip()

        elif text.startswith("stop "):

            entities["action"] = "stop"
            entities["target"] = text[5:].strip()

        elif (
            text == "screenshot"
            or text.startswith("screenshot ")
            or text.startswith("take a screenshot")
        ):

            entities["action"] = "screenshot"

        elif text == "mute":

            entities["action"] = "mute"

        elif text == "unmute":

            entities["action"] = "unmute"

        return entities

    # --------------------------------------------------------

    @staticmethod
    def _extract_memory_entities(
        text: str,
    ) -> dict[str, Any]:

        prefixes = (
            "remember that",
            "remember this",
            "don't forget",
            "do not forget",
            "save this",
            "keep this in mind",
        )

        for prefix in prefixes:

            if text.startswith(prefix):

                content = text[
                    len(prefix):
                ].strip()

                return {
                    "memory": content
                }

        return {}