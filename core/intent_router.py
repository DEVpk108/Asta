import re
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
        """Backward-compatible intent-only API."""
        return self.analyze(text).intent

    def analyze(self, text: str) -> IntentResult:
        if not text:
            return IntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                normalized_text="",
            )

        normalized = self._normalize(text)

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
                entities=self._extract_memory_entities(normalized),
                requires_memory=True,
                classifier="rules",
            )

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
                entities=self._extract_command_entities(normalized),
                requires_tools=True,
                classifier="rules",
            )

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

        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.20,
            normalized_text=normalized,
            classifier="rules",
        )

    @staticmethod
    def _normalize(text: str) -> str:
        # Speech-to-text commonly adds terminal punctuation. Keep the
        # normalized form stable so "screenshot." maps exactly like
        # "screenshot" while preserving useful characters inside commands.
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.!?,;:]+$", "", text)
        return text.strip()

    @staticmethod
    def _extract_command_entities(text: str) -> dict[str, Any]:
        for prefix, action in (
            ("open ", "open"),
            ("launch ", "launch"),
            ("start ", "start"),
            ("close ", "close"),
            ("run ", "run"),
            ("stop ", "stop"),
        ):
            if text.startswith(prefix):
                return {
                    "action": action,
                    "target": text[len(prefix):].strip(),
                }

        if text == "screenshot" or text.startswith("screenshot ") or text.startswith("take a screenshot"):
            return {"action": "screenshot"}

        if text == "mute":
            return {"action": "mute"}

        if text == "unmute":
            return {"action": "unmute"}

        return {}

    @staticmethod
    def _extract_memory_entities(text: str) -> dict[str, Any]:
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
                return {"memory": text[len(prefix):].strip()}

        return {}
