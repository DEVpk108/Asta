import re
from typing import Any

from .contracts.intent import (
    IntentResult,
    IntentType,
)


class IntentRouter:

    _COMMAND_PREFIXES = (
        ("open ", "open"),
        ("launch ", "launch"),
        ("start ", "start"),
        ("close ", "close"),
        ("run ", "run"),
        ("stop ", "stop"),
    )

    _COMMAND_LEADS = (
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
        "i want you to ",
        "i need you to ",
        "okay ",
        "ok ",
        "hey ",
    )

    _NON_MEMORY_REQUEST_PREFIXES = (
        "tell me ",
        "give me ",
        "make ",
        "show me ",
        "explain ",
        "describe ",
        "what ",
        "who ",
        "how ",
        "why ",
        "when ",
        "where ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
        "please ",
        "i want ",
        "i need ",
    )

    _COMPOUND_SEPARATORS = (
        ", then ",
        " and then ",
        " then ",
        " after that ",
        " followed by ",
        " and ",
    )

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
            memory_text = self._extract_memory_entities(normalized).get("memory", "")
            if memory_text and not self._contains_follow_up_request(memory_text):
                return IntentResult(
                    intent=IntentType.MEMORY,
                    confidence=0.98,
                    normalized_text=normalized,
                    entities={"memory": memory_text},
                    requires_memory=True,
                    classifier="rules",
                )

        compound_commands = self._extract_compound_commands(normalized)
        if compound_commands:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities={"commands": compound_commands},
                requires_tools=True,
                classifier="rules",
            )

        command_entities = self._extract_command_entities(normalized)
        if command_entities:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities=command_entities,
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
        # normalized form stable while allowing natural chatter around commands.
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.!?,;:]+$", "", text)
        return text.strip()

    @classmethod
    def _extract_compound_commands(cls, text: str) -> list[dict[str, Any]]:
        """Parse simple sequential commands joined by natural separators."""
        parts = [text]
        for separator in cls._COMPOUND_SEPARATORS:
            if separator in text:
                parts = [part.strip(" ,") for part in text.split(separator)]
                break

        if len(parts) < 2:
            return []

        commands = []
        for part in parts:
            command = cls._extract_command_entities(part)
            if not command or "commands" in command:
                return []
            commands.append(command)

        return commands if len(commands) >= 2 else []

    @classmethod
    def _extract_command_entities(cls, text: str) -> dict[str, Any]:
        # Direct commands: "open chrome", "take a screenshot", etc.
        direct = cls._extract_direct_command(text)
        if direct:
            return direct

        # Natural wrappers: "please open chrome", "can you open chrome",
        # "okay, open chrome", and similar voice-assistant phrasing.
        stripped = text
        changed = True
        while changed:
            changed = False
            for lead in cls._COMMAND_LEADS:
                if stripped.startswith(lead):
                    stripped = stripped[len(lead):].lstrip(" ,")
                    changed = True
                    break

        direct = cls._extract_direct_command(stripped)
        if direct:
            return direct

        # Embedded command: "nothing else, open chrome" or
        # "hey asta, please open chrome". Search for the command boundary,
        # but only accept an explicit executable action phrase.
        pattern = re.compile(
            r"(?:^|[\s,;:])"
            r"(?:(?:please|can you|could you|would you|will you|okay|ok|hey)\s+)?"
            r"(?P<action>open|launch|start|close|run|stop)\s+"
            r"(?P<target>.+?)\s*$"
        )
        match = pattern.search(text)
        if match:
            target = match.group("target").strip(" ,.!?;:")
            if target:
                return {
                    "action": match.group("action"),
                    "target": target,
                }

        return {}

    @classmethod
    def _extract_direct_command(cls, text: str) -> dict[str, Any]:
        for prefix, action in cls._COMMAND_PREFIXES:
            if text.startswith(prefix):
                target = text[len(prefix):].strip(" ,.!?;:")
                if target:
                    return {
                        "action": action,
                        "target": target,
                    }

        if text in {"screenshot", "screen shot"}:
            return {"action": "screenshot"}

        if text.startswith("take a screenshot") or text.startswith("take a screen shot"):
            return {"action": "screenshot"}

        if text == "mute":
            return {"action": "mute"}

        if text == "unmute":
            return {"action": "unmute"}

        return {}

    @classmethod
    def _extract_memory_entities(cls, text: str) -> dict[str, Any]:
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

    @classmethod
    def _contains_follow_up_request(cls, memory_text: str) -> bool:
        """Avoid consuming a mixed utterance when it contains a later request."""
        normalized = cls._normalize(memory_text)
        if not normalized:
            return False

        for prefix in cls._NON_MEMORY_REQUEST_PREFIXES:
            if prefix in normalized:
                return True

        # A second imperative sentence is commonly produced by speech
        # recognition as a single utterance. Treat it as a conversational /
        # unknown request so the LLM can interpret the full message.
        if re.search(
            r"\b(?:tell|give|show|make|explain|describe|ask|play|write|say)\s+me\b",
            normalized,
        ):
            return True

        return False
