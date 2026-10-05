import re
from typing import Any

from .contracts.intent import (
    IntentResult,
    IntentType,
)
from .media import parse_media_request
from .transliteration import canonicalize_command, strip_wake_remnant


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

    _DANGLING_CLAUSE_PATTERN = re.compile(
        r"(?<=\S)\s*,?\s+(?:and then|and|then)\s+"
        r"(?:search(?:\s+for)?|look\s+up|open|launch|start|close|play|type)"
        r"(?:\s+(?:for|the|a|about))?$",
        re.IGNORECASE,
    )

    _COMMA_COMMAND_PATTERN = re.compile(
        r",\s*(?=(?:please\s+|can you\s+|could you\s+|would you\s+|will you\s+)?"
        r"(?:open|launch|start|close|run|stop|take|capture|screenshot|mute|unmute|search|look\s+up)\b)",
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

    _VISUAL_VERIFICATION_SUFFIX_PATTERN = re.compile(
        r"^(?P<command>.+?)\s+and\s+(?P<verification>"
        r"(?:verify|check)\s+.+)$",
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
        # Voice transcripts can carry a merged wake word ("upyasta open
        # chrome") or Hindi/Devanagari command grammar ("क्रोम खोलो").
        normalized = self._normalize(
            canonicalize_command(strip_wake_remnant(normalized))
        )
        # Drop a dangling clause the STT cut off ("open chrome and search
        # for"), so the complete first command still runs.
        normalized = self._DANGLING_CLAUSE_PATTERN.sub("", normalized).strip()

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

        transport = self._extract_transport_command(normalized)
        if transport:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.97,
                normalized_text=normalized,
                entities=transport,
                requires_tools=True,
                classifier="rules",
            )

        screen_question = self._extract_screen_question(normalized)
        if screen_question:
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.96,
                normalized_text=normalized,
                entities=screen_question,
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

        command_entities = self._refine_command_entities(
            self._extract_command_entities(
                normalized,
                known_providers=self._media_providers,
            )
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

        recovered_media = self._refine_command_entities(
            self._recover_media_command(normalized)
        )
        # "start a timer" is not a media request; only treat a bare "start X"
        # as playback when a media provider is named explicitly.
        if (
            recovered_media
            and normalized.startswith("start ")
            and not recovered_media.get("provider")
        ):
            recovered_media = {}
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
        prefixes = (
            "hey asta, ",
            "hey asta ",
            "hello asta, ",
            "hello asta ",
            "wake up asta, ",
            "wake up asta ",
        )
        for prefix in prefixes:
            if value.startswith(prefix):
                return value[len(prefix):].strip()
        return value

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.strip().lower()
        # Spoken fillers ("can you uh open chrome") break command grammar.
        text = re.sub(r"(?:^|(?<=[\s,]))(?:uh+|um+|uhm|erm?|hmm+|ah+)(?=[\s,.!?]|$)[,]?", " ", text)
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

    _TRANSPORT_LEAD = re.compile(
        r"^(?:(?:okay|ok|uk|hey|hi|please|so|and|now|asta|just|uh|um|"
        r"can you|could you|would you|will you)[\s,]+)+"
    )
    _ITEM = r"(?:(?:the|this|that|my|current)\s+)?(?:song|track|music|gaana|gana|one|playback|audio|it)"
    _APP_SUFFIX = re.compile(r"\s+(?:on|in)\s+(spotify|apple music|youtube music|youtube)$|\s+(spotify)$")
    _TRANSPORT_PATTERNS = (
        ("pause", re.compile(
            rf"^(?:pause|paus|pauze|pose|paws)(?:\s+{_ITEM})?$|"
            rf"^(?:stop|hold)\s+{_ITEM}$|"
            r"^(?:gaana|gana|song|music)\s+(?:rok|roko|ruko|band|bandh)(?:\s+(?:do|karo|kar do|kardo))?$"
        )),
        ("play", re.compile(
            rf"^(?:resume|unpause|continue)(?:\s+{_ITEM})?(?:\s+playing)?$|"
            r"^(?:play|start)\s+(?:it\s+)?again$|^(?:gaana|gana)\s+(?:chalao|chala do|shuru karo)$"
        )),
        ("next", re.compile(
            rf"^(?:next|skip|change|switch)(?:\s+{_ITEM})?(?:\s+please)?$|"
            r"^(?:play|go to|put on)\s+(?:the\s+)?next(?:\s+(?:song|track|one))?$|"
            r"^(?:play|put on)\s+(?:(?:a|an|some|any)\s+)?(?:different|another|other|new|else)\s+(?:song|track|one)$|"
            r"^(?:agla|agla wala|next)\s+(?:gaana|gana|song)(?:\s+(?:chalao|lagao|bajao))?$|"
            r"^(?:gaana|gana|song)\s+(?:badlo|badal do|change karo)$"
        )),
        ("previous", re.compile(
            r"^(?:previous|prev)(?:\s+(?:song|track|one))?$|"
            r"^(?:play|go to|go back to)\s+(?:the\s+)?previous(?:\s+(?:song|track|one))?$|"
            r"^go\s+back(?:\s+(?:a|one)\s+(?:song|track))?$|"
            r"^(?:go back to|back to)\s+(?:the\s+)?last\s+(?:song|track)$|"
            r"^(?:pichla|pichhla|previous)\s+(?:gaana|gana|song)(?:\s+(?:chalao|lagao|bajao))?$"
        )),
        ("now_playing", re.compile(
            r"^(?:what|which)(?:'s|\s+is)?\s+(?:song|track|music)\s+(?:is\s+)?(?:this|playing|that|on)(?:\s+(?:right\s+)?now)?$|"
            r"^what(?:'s|\s+is)\s+(?:playing|this song|this track|the song|the current song)(?:\s+(?:right\s+)?now)?$|"
            r"^(?:who\s+(?:sings|sang)\s+(?:this|that)(?:\s+song)?|what(?:'s|\s+is)\s+the\s+name\s+of\s+(?:this|the)\s+song)$|"
            r"^(?:kaun\s+sa|konsa)\s+(?:gaana|gana)\s+(?:hai|chal raha hai)$"
        )),
    )

    @classmethod
    def _extract_transport_command(cls, text: str) -> dict[str, Any]:
        """Pause / resume / next / previous / what's playing, no vision needed."""
        value = cls._TRANSPORT_LEAD.sub("", str(text or "").strip().lower())
        value = re.sub(r"[.!?,]+", " ", value)
        value = re.sub(r"\b(?:uh+|um+|erm|hmm)\b", " ", value)
        value = re.sub(r"\s+(?:please|now|right now|for me)$", "", " ".join(value.split()))
        provider = None
        suffix = cls._APP_SUFFIX.search(value)
        if suffix:
            provider = (suffix.group(1) or suffix.group(2)).replace(" ", "_")
            value = value[: suffix.start()].strip()
        if not value:
            return {}
        for operation, pattern in cls._TRANSPORT_PATTERNS:
            if pattern.match(value):
                entities = {"action": "media", "operation": operation}
                if provider == "spotify":
                    entities["provider"] = provider
                return entities
        return {}

    _SCREEN_QUESTION = re.compile(
        r"^(?:(?:hey|okay|ok|so|and|now|please)[\s,]+)*(?:"
        r"(?:can|could)\s+you\s+(?:see|tell\s+(?:me\s+)?what(?:'s|\s+is)\s+on)\b|"
        r"(?:(?:can|could)\s+you\s+)?(?:find|spot|read)\s+.*\bon\s+(?:my|the|this)\s+screen\b|"
        r"do\s+you\s+see\b|what\s+do\s+you\s+see\b|"
        r"what(?:'s|\s+is)\s+on\s+(?:my|the|this)\s+screen\b|"
        r"(?:look|looking)\s+at\s+(?:my|the|this)\s+screen\b|"
        r"(?:(?:can|could)\s+you\s+)?(?:describe|read|check)\s+(?:my|the|this)\s+screen\b|"
        r"is\s+there\s+.+\s+on\s+(?:my|the)\s+screen\b)",
        re.IGNORECASE,
    )
    SCREEN_QUESTION_PREFIX = "Answer the user's question about the current screenshot"

    @classmethod
    def _extract_screen_question(cls, text: str) -> dict[str, Any]:
        """Questions about the screen ("can you see the search field?")."""
        value = str(text or "").strip()
        if not value or not cls._SCREEN_QUESTION.match(value):
            return {}
        return {
            "action": "inspect",
            "tool": "vision.inspect",
            "prompt": (
                f"{cls.SCREEN_QUESTION_PREFIX}: \"{value}\". Put a short, "
                "direct spoken answer (one or two sentences) in summary. Set "
                "visual_match=true when the answer is yes or the asked item "
                "is visible."
            ),
        }

    @classmethod
    def _extract_compound_commands(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> list[dict[str, Any]]:
        """Parse sequential commands joined by explicit or natural separators."""
        media_sequence = cls._extract_open_media_sequence(
            text,
            known_providers=known_providers,
        )
        if media_sequence:
            return media_sequence
        visual = cls._VISUAL_VERIFICATION_SUFFIX_PATTERN.match(text)
        if visual:
            first_text = visual.group("command").strip()
            verification = visual.group("verification").strip()
            first = cls._extract_command_entities(
                first_text,
                known_providers=known_providers,
            )
            if first:
                target = str(first.get("target") or "").strip()
                condition = verification
                if target and re.search(r"\bit\b", condition, re.IGNORECASE):
                    condition = re.sub(
                        r"\bit\b",
                        target,
                        condition,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                return [
                    first,
                    {
                        "action": "inspect",
                        "tool": "vision.inspect",
                        "prompt": (
                            "Verify the following condition from the current "
                            f"screenshot: {condition}."
                        ),
                    },
                ]

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
            visual = cls._VISUAL_VERIFICATION_SUFFIX_PATTERN.match(part)
            if visual:
                first_text = visual.group("command").strip()
                verification = visual.group("verification").strip()

                first = cls._extract_command_entities(
                    first_text,
                    known_providers=known_providers,
                )
                if first:
                    target = str(first.get("target") or "").strip()
                    condition = verification
                    if target and re.search(r"\bit\b", condition, re.IGNORECASE):
                        condition = re.sub(
                            r"\bit\b",
                            target,
                            condition,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                    prompt = (
                        "Verify the following condition from the current "
                        f"screenshot: {condition}."
                    )
                    commands.extend(
                        (
                            first,
                            {
                                "action": "inspect",
                                "tool": "vision.inspect",
                                "prompt": prompt,
                            },
                        )
                    )
                    continue

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
    def _extract_open_media_sequence(
        cls,
        text: str,
        *,
        known_providers=None,
    ) -> list[dict[str, Any]]:
        """Parse a generic "open provider, search for query, and play it" flow.

        Provider names come from the registered media providers; no application
        name is hard-coded here.
        """
        match = re.fullmatch(
            r"(?:please\s+)?(?:open|launch|start)\s+"
            r"(?P<target>[^,]+?)\s*,\s*"
            r"search\s+for\s+(?P<query>.+?)\s*,?\s*"
            r"(?:and\s+)?play(?:\s+(?:it|that|this))?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return []

        target = match.group("target").strip(" ,.!?;:")
        query = match.group("query").strip(" ,.!?;:")
        if not target or not query:
            return []

        normalized_target = cls._normalize(target)
        providers = {
            cls._normalize(str(name))
            for name in (known_providers or ())
            if str(name).strip()
        }
        provider = normalized_target if normalized_target in providers else None
        if provider is None:
            return []

        return [
            {
                "action": "open",
                "target": target,
            },
            {
                "action": "media",
                "operation": "play",
                "query": query,
                "provider": provider,
            },
        ]

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
        # Only accept an action near the start of the utterance. Searching the
        # whole sentence turned questions such as "tell me how to open excel
        # files" into an "open" command.
        if match and len(text[:match.start()].split()) > 2:
            match = None
        if match:
            target = match.group("target").strip(" ,.!?;:")
            if target:
                return {
                    "action": match.group("action"),
                    "target": target,
                }

        return {}

    # Targets that make "run"/"start" conversational rather than a process.
    _NON_EXECUTABLE_TARGET_LEADS = (
        "me ", "us ", "through ", "by ", "over ", "into ", "away", "out ",
        "a ", "an ", "some ", "with ", "from ", "again", "it again",
    )
    _MEDIA_STOP_TARGETS = {
        "music", "the music", "song", "the song", "this song", "playback",
        "the playback", "video", "the video", "audio", "the audio", "track",
        "the track", "playing", "the player", "podcast", "the podcast",
    }
    _MEDIA_TRANSPORT_QUERY_WORDS = {
        "", "song", "track", "one", "video", "music", "episode", "please",
        "the song", "the track", "the video", "the music", "the episode",
        "this song", "this track", "this video", "song please", "track please",
    }
    _RESUME_QUERY = re.compile(
        r"^(?:it|that|this|again|music|some\s+music|my\s+music|something|"
        r"(?:that|this|the|my)\s+(?:song|track|music)(?:\s+again)?|"
        r"(?:the\s+|my\s+)?(?:last|previous|recent)\s+(?:song|track|music|one)\b.*|"
        r"what(?:ever)?\s+i\s+was\s+(?:listening|playing)\b.*)$",
        re.IGNORECASE,
    )

    _TRAILING_POLITENESS = re.compile(
        r"(?:\s*,?\s+(?:for me|for us|please|right now|now))+$"
    )

    _VAGUE_LEAD = re.compile(
        r"^(?:(?:something|anything|some|any|a|an)\s+)?(?:(?:like|kind of|type of|sort of|of)\s+)?"
    )
    _VAGUE_TAIL = re.compile(r"\s+(?:type|kind|style|vibes?|sort|types)$")

    @classmethod
    def _clean_vague_media_query(cls, query: str) -> str:
        """"something egyptian music type" -> "egyptian music"."""
        value = " ".join(str(query or "").split())
        cleaned = cls._VAGUE_TAIL.sub("", cls._VAGUE_LEAD.sub("", value)).strip()
        return cleaned if cleaned and len(cleaned) >= 3 else value

    @classmethod
    def _refine_command_entities(cls, entities: dict[str, Any]) -> dict[str, Any]:
        """Reject or correct common voice misroutes before tool selection."""
        if not entities:
            return entities

        action = str(entities.get("action") or "")

        if action == "search":
            query = cls._TRAILING_POLITENESS.sub("", str(entities.get("query") or "").strip()).strip(" ,.!?;:")
            if not query:
                return {}
            refined = dict(entities)
            refined["query"] = query
            return refined

        if action == "media":
            operation = str(entities.get("operation") or "").lower()
            query = str(entities.get("query") or "").strip().lower()
            provider = str(entities.get("provider") or "")
            if " on " in f" {provider} ":
                # "eppal on apple music": keep the last named app.
                entities = {**entities, "provider": provider.rsplit(" on ", 1)[-1].strip()}
            if operation == "play" and query:
                from core.media.artists import artist_request, snap_artist_names

                # "some udit narayan songs" means the artist's songs, not a
                # track literally called that; also repair glued names.
                snapped = snap_artist_names(query)
                artist = artist_request(snapped)
                if artist:
                    return {**entities, "query": artist, "artist": artist}
                if snapped != query:
                    entities = {**entities, "query": snapped}
                    query = snapped
                vague = cls._clean_vague_media_query(query)
                if vague and vague != query:
                    entities = {**entities, "query": vague}
                    query = vague
            if operation == "play" and cls._RESUME_QUERY.match(query):
                # "play the last song I was listening to" / "play that song":
                # resume the player instead of searching for those words.
                refined = {k: v for k, v in entities.items() if k != "query"}
                return refined
            # "next question" is conversation, not a media skip.
            if (
                operation in {"next", "previous", "pause", "resume", "stop", "toggle"}
                and query not in cls._MEDIA_TRANSPORT_QUERY_WORDS
                and not entities.get("provider")
            ):
                return {}
            return entities

        target = entities.get("target")
        if action not in {"open", "launch", "start", "close", "run", "stop"} or not isinstance(target, str):
            return entities

        cleaned = cls._TRAILING_POLITENESS.sub("", target.strip()).strip(" ,.!?;:")
        if action in {"open", "launch", "start"} and cleaned.startswith("up "):
            cleaned = cleaned[3:].strip()
        if not cleaned:
            return {}

        if action == "stop" and cleaned in cls._MEDIA_STOP_TARGETS:
            return {"action": "media", "operation": "pause"}

        if action in {"run", "start"} and (
            cleaned + " "
        ).startswith(cls._NON_EXECUTABLE_TARGET_LEADS):
            return {}

        refined = dict(entities)
        refined["target"] = cleaned
        if action == "launch":
            # The launch tool only knows PATH executables; "open" uses full
            # app discovery (Start menu, browser profiles, references).
            refined["action"] = "open"
        return refined

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

        search = cls._extract_search_command(text)
        if search:
            return search

        computer = cls._extract_computer_command(text)
        if computer:
            return computer

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
    def _extract_search_command(cls, text: str) -> dict[str, Any]:
        """Parse a generic GUI search request without naming a search engine."""
        value = cls._normalize(text)
        match = re.fullmatch(
            r"(?:search(?:\s+for)?|look\s+up)\s+"
            r"(?P<query>.+?)"
            r"(?:\s+(?:on|in|using)\s+(?P<target>[^,]+?))?",
            value,
            re.IGNORECASE,
        )
        if not match:
            return {}

        query = match.group("query").strip(" ,.!?;:")
        target = (match.group("target") or "").strip(" ,.!?;:")
        # "search for" with the query clipped off by STT must not type "for".
        if not query or query.lower() in {"for", "about", "on", "the", "a", "it", "something"}:
            return {}

        return {
            "action": "search",
            "query": query,
            **({"target": target} if target else {}),
        }

    @classmethod
    def _extract_computer_command(cls, text: str) -> dict[str, Any]:
        value = cls._normalize(text)

        double_click_prefixes = (
            "double click ",
            "double-click ",
            "doubleclick ",
        )
        for prefix in double_click_prefixes:
            if value.startswith(prefix):
                target = value[len(prefix):].strip(" ,.!?;:")
                if target:
                    return {
                        "action": "click",
                        "target": target,
                        "clicks": 2,
                    }

        right_click_prefixes = (
            "right click ",
            "right-click ",
            "rightclick ",
        )
        for prefix in right_click_prefixes:
            if value.startswith(prefix):
                target = value[len(prefix):].strip(" ,.!?;:")
                if target:
                    return {
                        "action": "click",
                        "target": target,
                        "button": "right",
                    }

        for prefix in ("click ", "click on "):
            if value.startswith(prefix):
                target = value[len(prefix):].strip(" ,.!?;:")
                if target:
                    return {
                        "action": "click",
                        "target": target,
                    }

        for prefix in ("type text ", "type ", "write "):
            if value.startswith(prefix):
                text_value = value[len(prefix):].strip()
                if text_value:
                    return {
                        "action": "type_text",
                        "text": text_value,
                    }

        for prefix in ("press ", "press the "):
            if value.startswith(prefix):
                key = value[len(prefix):].strip(" ,.!?;:")
                if key:
                    return {
                        "action": "keypress",
                        "key": key,
                    }

        if value.startswith("scroll "):
            direction = value[len("scroll "):].strip()
            if direction in {"down", "lower"}:
                return {"action": "scroll", "amount": -5}
            if direction in {"up", "higher"}:
                return {"action": "scroll", "amount": 5}
            if direction in {"to top", "top"}:
                return {"action": "scroll", "amount": 100}
            if direction in {"to bottom", "bottom"}:
                return {"action": "scroll", "amount": -100}

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
