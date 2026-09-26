import re
from typing import Any

from .contracts.intent import (
    IntentResult,
    IntentType,
)
from .media import parse_media_request


class IntentRouter:

    def __init__(self, *, media_providers=None):
        self._media_providers = tuple(
            str(name).strip().lower()
            for name in (media_providers or ())
            if str(name).strip()
        )

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
        normalized = self._strip_wakeword_prefix(normalized)

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

        note_entities = self._extract_note_command(normalized)
        if note_entities:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities=note_entities,
                requires_tools=True,
                classifier="rules",
            )

        compound_commands = self._extract_compound_commands(
            normalized,
            known_providers=self._media_providers,
        )
        if compound_commands:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities={"commands": compound_commands},
                requires_tools=True,
                classifier="rules",
            )

        command_entities = self._extract_command_entities(
            normalized,
            known_providers=self._media_providers,
        )
        if command_entities:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.98,
                normalized_text=normalized,
                entities=command_entities,
                requires_tools=True,
                classifier="rules",
            )

        recovered_media = self._recover_media_command(normalized)
        if recovered_media:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.94,
                normalized_text=normalized,
                entities=recovered_media,
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
    def _strip_wakeword_prefix(text: str) -> str:
        """Remove a wake phrase already consumed by the voice runtime."""
        value = str(text or "").strip()
        patterns = (
            r"^(?:hey|hello)\\s+asta(?:\\s*[,;:]?\\s+|\\s+)(?=\\S)",
            r"^wake\\s+up\\s+asta(?:\\s*[,;:]?\\s+|\\s+)(?=\\S)",
        )
        for pattern in patterns:
            value = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE).strip()
            if value != str(text or "").strip():
                return value
        return value

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.!?,;:]+$", "", text)
        return text.strip()

    @classmethod
    def _extract_note_command(cls, text: str) -> dict[str, Any]:
        normalized = cls._normalize(text)

        list_phrases = {
            "list notes",
            "show notes",
            "show my notes",
            "list my notes",
            "what are my notes",
        }
        if normalized in list_phrases:
            return {"action": "list_notes"}

        for prefix in (
            "search notes for ",
            "search my notes for ",
            "find notes about ",
            "find my notes about ",
        ):
            if normalized.startswith(prefix):
                query = normalized[len(prefix):].strip()
                if query:
                    return {"action": "search_notes", "query": query}

        for prefix in (
            "read note ",
            "read my note ",
            "open note ",
            "open my note ",
            "show note ",
            "show my note ",
        ):
            if normalized.startswith(prefix):
                target = normalized[len(prefix):].strip()
                if target:
                    return {"action": "read_note", "target": target}

        create_prefixes = (
            "take a note ",
            "take a note:",
            "take note ",
            "take note:",
            "write a note ",
            "write a note:",
            "write a new note ",
            "write a new note:",
            "create a note ",
            "create a note:",
            "create a new note ",
            "create a new note:",
            "make a note ",
            "make a note:",
            "make a new note ",
            "make a new note:",
            "save a note ",
            "save a note:",
        )
        for prefix in create_prefixes:
            if not normalized.startswith(prefix):
                continue

            payload = normalized[len(prefix):].strip(" :,-")
            if not payload:
                return {}

            title = None
            content = payload
            for lead in ("titled ", "called "):
                if payload.startswith(lead):
                    remainder = payload[len(lead):].strip()
                    split_at = remainder.find(" saying ")
                    if split_at > 0:
                        title = remainder[:split_at].strip()
                        content = remainder[split_at + len(" saying "):].strip()
                    else:
                        split_at = remainder.find(" with content ")
                        if split_at > 0:
                            title = remainder[:split_at].strip()
                            content = remainder[split_at + len(" with content "):].strip()
                    break

            if not content:
                return {}

            result = {"action": "create_note", "content": content}
            if title:
                result["title"] = title
            return result

        return {}

    @classmethod
    def _extract_compound_commands(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> list[dict[str, Any]]:
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
            # Spoken commands sometimes omit "and" before a screenshot phrase,
            # e.g. "open camera take screenshot". Check this before the generic
            # open/close parser so the suffix does not get swallowed into target.
            implicit = cls._IMPLICIT_SCREENSHOT_SUFFIX_PATTERN.match(part)
            if implicit:
                first = cls._extract_command_entities(
                    implicit.group("command").strip(),
                    known_providers=known_providers,
                )
                second = cls._extract_direct_command(
                    implicit.group("screenshot").strip(),
                    known_providers=known_providers,
                )
                if first and second:
                    commands.extend((first, second))
                    continue

            direct = cls._extract_command_entities(
                part,
                known_providers=known_providers,
            )
            if direct and "commands" not in direct:
                commands.append(direct)
                continue

            return []

        return commands if len(commands) >= 2 else []

    @classmethod
    def _extract_command_entities(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> dict[str, Any]:
        direct = cls._extract_direct_command(
            text,
            known_providers=known_providers,
        )
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

        direct = cls._extract_direct_command(
            stripped,
            known_providers=known_providers,
        )
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
    def _extract_direct_command(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> dict[str, Any]:
        for prefix, action in cls._COMMAND_PREFIXES:
            if text.startswith(prefix):
                target = text[len(prefix):].strip(" ,.!?;:")
                if target:
                    return {
                        "action": action,
                        "target": target,
                    }

        # Whisper/STT can occasionally collapse an action and its target
        # into one token, for example "OpenSpotify" -> "openspotify".
        # Accept that form conservatively without depending on any
        # specific application name.
        compact_match = re.fullmatch(
            r"(open|launch|start|close|run|stop)([a-z0-9][a-z0-9._-]*)",
            text,
        )
        if compact_match:
            action, target = compact_match.groups()
            if len(target) >= 3 or target in {"it", "this", "that"}:
                return {
                    "action": action,
                    "target": target,
                }

        media = cls._extract_media_command(
            text,
            known_providers=known_providers,
        )
        if media:
            return media

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
    def _extract_media_command(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> dict[str, Any]:
        request = parse_media_request(
            text,
            known_providers=known_providers,
        )
        return request.to_entities() if request is not None else {}

    def _recover_media_command(self, text: str) -> dict[str, Any]:
        request = parse_media_request(
            text,
            known_providers=self._media_providers,
        )
        return request.to_entities() if request is not None else {}

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
