from typing import Any

from .contracts.intent import (
    IntentResult,
    IntentType,
)


# ============================================================
# Intent Router
# ============================================================

class IntentRouter:

    def route(self, text: str) -> IntentType:
        """
        Backward-compatible API.

        Existing ASTA code can continue using:

            router.route(text)

        New code should prefer:

            router.analyze(text)
        """

        return self.analyze(text).intent

    # --------------------------------------------------------
    # Main analysis
    # --------------------------------------------------------

    def analyze(self, text: str) -> IntentResult:

        if not text:

            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                normalized_text="",
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
                requires_memory=True,
                classifier="rules",
            )

        # ----------------------------------------------------
        # Commands
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
                requires_tools=True,
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

        if normalized.startswith(
            conversation_phrases
        ):

            return IntentResult(
                intent=IntentType.CONVERSATION,
                confidence=0.98,
                normalized_text=normalized,
                classifier="rules",
            )

        # ----------------------------------------------------
        # Unknown for now
        #
        # Future:
        #
        # UNKNOWN
        #    ↓
        # Small intent model
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
    # Command entities
    # ========================================================

    @staticmethod
    def _extract_command_entities(
        text: str,
    ) -> dict[str, Any]:

        if text.startswith("open "):
            return {
                "action": "open",
                "target": text[5:].strip(),
            }

        if text.startswith("launch "):
            return {
                "action": "launch",
                "target": text[7:].strip(),
            }

        if text.startswith("start "):
            return {
                "action": "start",
                "target": text[6:].strip(),
            }

        if text.startswith("close "):
            return {
                "action": "close",
                "target": text[6:].strip(),
            }

        if text.startswith("run "):
            return {
                "action": "run",
                "target": text[4:].strip(),
            }

        if text.startswith("stop "):
            return {
                "action": "stop",
                "target": text[5:].strip(),
            }

        if (
            text == "screenshot"
            or text.startswith("screenshot ")
            or text.startswith("take a screenshot")
        ):
            return {
                "action": "screenshot",
            }

        if text == "mute":
            return {
                "action": "mute",
            }

        if text == "unmute":
            return {
                "action": "unmute",
            }

        return {}

    # ========================================================
    # Memory entities
    # ========================================================

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