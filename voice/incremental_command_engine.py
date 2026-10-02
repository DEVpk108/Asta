from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.applications.manager import (
    _abbreviation_match,
    normalize_application_name,
)


@dataclass(frozen=True, slots=True)
class IncrementalCommit:
    """A semantic command committed before the user finishes speaking."""

    commit_id: str
    text: str
    entities: dict[str, Any]
    source_text: str


class IncrementalCommandDetector:
    """Detect safe, semantically complete action prefixes from partial STT.

    V1 deliberately commits only application-open actions. Other commands are
    left for the normal end-of-utterance path so destructive or context-heavy
    actions do not fire from an unstable partial transcript.
    """

    EARLY_ACTIONS = frozenset({"open", "launch", "start"})
    # Only remove discourse padding from "open up X". Keep articles such as
    # "the" because the application resolver may use them as part of a name.
    TARGET_FILLER = frozenset({"up"})
    DISQUALIFYING_TARGET_WORDS = frozenset(
        {
            "and",
            "then",
            "after",
            "once",
            "when",
            "there",
            "here",
            "for",
            "me",
            "please",
            "can",
            "could",
            "would",
            "will",
        }
    )

    # Words that may precede the action at the start of an utterance. The
    # action itself must come first (after these); "open"/"start" in the
    # middle of a sentence such as "tell me how to open excel files" is not a
    # command.
    LEADING_FILLER_PHRASES = (
        ("hey", "asta"),
        ("hello", "asta"),
        ("wake", "up", "asta"),
        ("asta",),
        ("can", "you"),
        ("could", "you"),
        ("would", "you"),
        ("will", "you"),
        ("go", "ahead", "and"),
        ("please",),
        ("okay",),
        ("ok",),
        ("hey",),
        ("so",),
        ("now",),
        ("just",),
    )
    # A later command may follow one of these connectives once an earlier
    # action in the same utterance has already been committed.
    CONNECTIVES = frozenset({"and", "then", "also"})
    # Tokens that can never identify an application on their own. They match
    # inside many application names ("the" in "Weather", "new" in "News").
    WEAK_TARGET_TOKENS = frozenset(
        {"a", "an", "the", "my", "this", "that", "it", "some", "new", "of", "to", "in", "on"}
    )
    LEADING_ARTICLES = frozenset({"a", "an", "the", "my"})
    # "the notes app" should match "Notes".
    GENERIC_APP_WORDS = frozenset({"app", "application", "program", "software"})

    def __init__(
        self,
        *,
        application_manager=None,
        stable_updates: int = 2,
    ):
        self.application_manager = application_manager
        self.stable_updates = max(1, int(stable_updates))
        self._candidate_signature: tuple[str, str] | None = None
        self._candidate_hits = 0
        self._commits: list[IncrementalCommit] = []
        self._commit_counter = 0

    @staticmethod
    def _normalize_preserve_case(text: str) -> str:
        value = " ".join(str(text or "").strip().split())
        return value.rstrip(" .!?;:,")

    @classmethod
    def _normalize(cls, text: str) -> str:
        return cls._normalize_preserve_case(text).lower()

    @classmethod
    def _clean_target(cls, words: list[str]) -> str:
        cleaned = list(words)
        while cleaned and cleaned[0] in cls.TARGET_FILLER:
            cleaned.pop(0)
        return " ".join(cleaned).strip()

    @classmethod
    def _strip_leading_fillers(cls, words: list[str]) -> int:
        """Return the index of the first word after leading filler phrases."""
        index = 0
        changed = True
        while changed and index < len(words):
            changed = False
            for phrase in cls.LEADING_FILLER_PHRASES:
                end = index + len(phrase)
                if tuple(words[index:end]) == phrase:
                    index = end
                    changed = True
                    break
        return index

    def _allowed_action_positions(self, words: list[str]) -> set[int]:
        positions = {self._strip_leading_fillers(words)}
        if self._commits:
            for index, word in enumerate(words[:-1]):
                if word in self.CONNECTIVES:
                    positions.add(index + 1 + self._strip_leading_fillers(words[index + 1:]))
        return positions

    @classmethod
    def _is_confident_application_match(cls, target: str, app_name: str) -> bool:
        """Require a whole-word match, not a substring/fuzzy similarity.

        ApplicationManager.resolve() is intentionally forgiving for explicit
        commands, but an early commit fires while the user is still talking,
        so it must only accept unambiguous names.
        """
        query_tokens = [
            token
            for token in normalize_application_name(target).split()
            if token not in cls.GENERIC_APP_WORDS
        ]
        while query_tokens and query_tokens[0] in cls.LEADING_ARTICLES:
            query_tokens.pop(0)
        name_tokens = normalize_application_name(app_name).split()
        if not query_tokens or not name_tokens:
            return False
        if all(token in cls.WEAK_TARGET_TOKENS for token in query_tokens):
            return False
        if query_tokens == name_tokens:
            return True

        strong = True
        for token in query_tokens:
            if token in name_tokens:
                continue
            if len(token) >= 4 and any(name.startswith(token) for name in name_tokens):
                continue
            strong = False
            break
        if strong:
            return True

        # Compact abbreviations such as "vscode" -> "Visual Studio Code".
        compact = "".join(query_tokens)
        return len(compact) >= 4 and _abbreviation_match(query_tokens, name_tokens)

    @staticmethod
    def _target_has_disqualifying_words(target: str) -> bool:
        return any(
            token in IncrementalCommandDetector.DISQUALIFYING_TARGET_WORDS
            for token in target.split()
        )

    def _resolve_application(self, target: str):
        manager = self.application_manager
        if manager is None:
            return None

        try:
            return manager.resolve(target)
        except Exception:
            return None

    def _find_open_candidate(self, transcript: str):
        words = self._normalize(transcript).split()
        if len(words) < 2:
            return None

        allowed_positions = self._allowed_action_positions(words)
        for action_index, word in enumerate(words):
            if word not in self.EARLY_ACTIONS:
                continue
            if action_index not in allowed_positions:
                continue

            for end in range(action_index + 1, len(words) + 1):
                target_words = words[action_index + 1 : end]
                if not target_words:
                    continue

                target = self._clean_target(target_words)
                if not target:
                    continue

                # Do not allow discourse/connective words to become part of an
                # application name, e.g. "open chrome and once...".
                if self._target_has_disqualifying_words(target):
                    continue

                application = self._resolve_application(target)
                if application is None:
                    continue

                app_name = str(getattr(application, "name", "") or target).strip()
                if not app_name:
                    continue
                if not self._is_confident_application_match(target, app_name):
                    continue

                source_text = " ".join(
                    words[action_index:end]
                ).strip()
                canonical = f"{word} {app_name}".strip()
                signature = (word, app_name.casefold())
                committed_signatures = {
                    (
                        str(commit.entities.get("action") or "").strip().lower(),
                        str(commit.entities.get("target") or "").casefold(),
                    )
                    for commit in self._commits
                }
                if signature in committed_signatures:
                    continue
                return signature, canonical, source_text

        return None

    def observe(self, transcript: str) -> IncrementalCommit | None:
        normalized = self._normalize(transcript)
        candidate = self._find_open_candidate(normalized)

        if candidate is None:
            self._candidate_signature = None
            self._candidate_hits = 0
            return None

        signature, canonical, source_text = candidate
        if signature == self._candidate_signature:
            self._candidate_hits += 1
        else:
            self._candidate_signature = signature
            self._candidate_hits = 1

        if self._candidate_hits < self.stable_updates:
            return None

        self._commit_counter += 1
        action, _app_name_normalized = signature
        commit = IncrementalCommit(
            commit_id=f"inc-{self._commit_counter}",
            text=canonical,
            entities={
                "action": action,
                "target": canonical[len(action) :].strip(),
            },
            source_text=source_text,
        )
        self._commits.append(commit)
        self._candidate_signature = None
        self._candidate_hits = 0
        return commit

    @property
    def commits(self) -> tuple[IncrementalCommit, ...]:
        return tuple(self._commits)

    @classmethod
    def _strip_follow_up_leads(cls, text: str) -> str:
        # Preserve the user's casing for the final remainder while matching
        # discourse prefixes case-insensitively.
        value = cls._normalize_preserve_case(text)
        if not value:
            return ""

        # Trailing politeness from the committed command ("open chrome for
        # me and ...") is not part of the follow-up instruction.
        value = re.sub(
            r"^(?:(?:for\s+me|for\s+us|please)\s*,?\s*)+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"^(?:(?:and|then|after that)\s+)+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"^(?:(?:once|when)\s+you(?:'re| are)\s+(?:there|here)\s*,?\s*)",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"^(?:(?:and|then)\s+)*(?:(?:can|could|would|will)\s+you\s+|please\s+|okay\s+|ok\s+)+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return value.strip(" ,.-")

    def finalize(self, transcript: str) -> str:
        """Return only the uncommitted tail of the final transcript."""
        # Search against a normalized lowercase copy, but slice from a
        # same-shape, case-preserving copy so names such as "Christopher Nolan"
        # remain intact for downstream intent parsing and UI display.
        original = self._normalize_preserve_case(transcript)
        value = original.lower()
        if not value or not self._commits:
            return transcript

        cursor = 0
        for commit in self._commits:
            markers = (
                self._normalize(commit.source_text),
                self._normalize(commit.text),
            )

            found = -1
            end = cursor
            for marker in markers:
                if not marker:
                    continue
                index = value.find(marker, cursor)
                if index >= 0:
                    found = index
                    end = index + len(marker)
                    break

            if found >= 0:
                cursor = end

        remainder = original[cursor:].strip()
        return self._strip_follow_up_leads(remainder)
