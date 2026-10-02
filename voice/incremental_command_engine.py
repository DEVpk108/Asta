from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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
    TARGET_FILLER = frozenset({"up", "the", "a", "an"})
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
    def _normalize(text: str) -> str:
        value = " ".join(str(text or "").strip().lower().split())
        return value.rstrip(" .!?;:,")

    @classmethod
    def _clean_target(cls, words: list[str]) -> str:
        cleaned = list(words)
        while cleaned and cleaned[0] in cls.TARGET_FILLER:
            cleaned.pop(0)
        return " ".join(cleaned).strip()

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

        for action_index, word in enumerate(words):
            if word not in self.EARLY_ACTIONS:
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
        value = cls._normalize(text)
        if not value:
            return ""

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
        value = self._normalize(transcript)
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

        remainder = value[cursor:].strip()
        return self._strip_follow_up_leads(remainder)
