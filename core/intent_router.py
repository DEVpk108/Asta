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

    _COMPOUND_SEPARATOR_PATTERN = re.compile(
        r"\s*(?:,\s*)?(?:and then|then|after that|followed by|and)\s+",
        re.IGNORECASE,
    )

    _COMMA_COMMAND_PATTERN = re.compile(
        r",\s*(?=(?:please\s+|can you\s+|could you\s+|would you\s+|will you\s+)?"
        r"(?:open|launch|start|close|run|stop|take|capture|screenshot|mute|unmute)\b)",
        re.IGNORECASE,
    )

    _IMPLICIT_SCREENSHOT_SUFFIX_PATTERN = re.compile(
        r"^(?P<command>.+?)\s+(?P<screenshot>"
        r"(?:take screenshot|take a screenshot|take screen shot|take a screen shot|"
        r"take the screenshot|take the screen shot|capture screenshot|capture a screenshot|"
        r"capture screen shot|capture a screen shot|capture the screenshot|"
        r"capture the screen shot|screenshot|screen shot))$",
        re.IGNORECASE,
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
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.!?,;:]+$", "", text)
        return text.strip()

    @classmethod
    def _extract_compound_commands(cls, text: str) -> list[dict[str, Any]]:
        """Parse sequential commands joined by explicit or natural separators."""
        first_parts = [
            part.strip(" ,")
            for part in cls._COMPOUND_SEPARATOR_PATTERN.split(text)
            if part.strip(" ,")
        ]

        parts: list[str] = []
        for part in first_parts:
            comma_parts = [
                sub.strip(" ,")
                for sub in cls._COMMA_COMMAND_PATTERN.split(part)
                if sub.strip(" ,")
            ]
            parts.extend(comma_parts)

        commands: list[dict[str, Any]] = []
        for part in parts:
            direct = cls._extract_command_entities(part)
            if direct and "commands" not in direct:
                commands.append(direct)
                continue

            # Spoken commands sometimes omit "and" before a screenshot phrase,
            # e.g. "open camera take screenshot". Split that suffix explicitly.
            implicit = cls._IMPLICIT_SCREENSHOT_SUFFIX_PATTERN.match(part)
            if implicit:
                first = cls._extract_command_entities(implicit.group("command").strip())
                second = cls._extract_direct_command(implicit.group("screenshot").strip())
                if first and second:
                    commands.extend((first, second))
                    continue

            return []

        return commands if len(commands) >= 2 else []

    @classmethod
    def _extract_command_entities(cls, text: str) -> dict[str, Any]:
        direct = cls._extract_direct_command(text)
        if direct:
            return direct

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

        screenshot_prefixes = (
            "take screenshot",
            "take a screenshot",
            "take screen shot",
            "take a screen shot",
            "take the screenshot",
            "take the screen shot",
            "capture screenshot",
            "capture a screenshot",
            "capture screen shot",
            "capture a screen shot",
            "capture the screenshot",
            "capture the screen shot",
        )
        if any(text.startswith(prefix) for prefix in screenshot_prefixes):
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
        normalized = cls._normalize(memory_text)
        if not normalized:
            return False

        for prefix in cls._NON_MEMORY_REQUEST_PREFIXES:
            if prefix in normalized:
                return True

        if re.search(
            r"\b(?:tell|give|show|make|explain|describe|ask|play|write|say)\s+me\b",
            normalized,
        ):
            return True

        return False
