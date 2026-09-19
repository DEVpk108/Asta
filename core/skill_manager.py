from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .contracts import IntentResult, IntentType, SkillDescriptor


SkillProvider = Callable[
    [IntentResult | None, str | None],
    tuple[SkillDescriptor, ...],
]


class SkillManager:
    """Own, discover and expose reusable reasoning skills for A.S.T.A."""

    def __init__(self, *, event_bus=None):
        self.event_bus = event_bus
        self._skills: dict[str, SkillDescriptor] = {}
        self._providers: dict[str, SkillProvider] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: SkillDescriptor) -> None:
        name = str(skill.name).strip()
        if not name:
            raise ValueError("skill name must be non-empty")
        if name in self._skills:
            raise ValueError(f"Skill already registered: {name}")

        with self._lock:
            self._skills[name] = skill

        self._emit("skill_registered", skill=skill.to_dict())

    def unregister(self, name: str) -> bool:
        key = str(name).strip()
        with self._lock:
            removed = self._skills.pop(key, None) is not None

        if removed:
            self._emit("skill_unregistered", skill=key)
        return removed

    def get(self, name: str) -> SkillDescriptor | None:
        with self._lock:
            return self._skills.get(str(name).strip())

    def list(self, *, enabled_only: bool = True) -> tuple[SkillDescriptor, ...]:
        with self._lock:
            values = tuple(self._skills.values())

        if enabled_only:
            values = tuple(skill for skill in values if skill.enabled)

        return values

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def register_provider(self, name: str, provider: SkillProvider) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("skill provider name must be non-empty")
        if not callable(provider):
            raise TypeError("skill provider must be callable")

        with self._lock:
            if key in self._providers:
                raise ValueError(f"Skill provider already registered: {key}")
            self._providers[key] = provider

        self._emit("skill_provider_registered", provider=key)

    def unregister_provider(self, name: str) -> bool:
        key = str(name).strip()
        with self._lock:
            removed = self._providers.pop(key, None) is not None

        if removed:
            self._emit("skill_provider_unregistered", provider=key)
        return removed

    def providers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._providers.keys())

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        intent: IntentResult | None = None,
        *,
        query: str | None = None,
        limit: int = 3,
    ) -> tuple[SkillDescriptor, ...]:
        limit = max(1, int(limit))
        query_value = str(query).strip().lower() if query else ""

        with self._lock:
            skills = list(self._skills.values())
            providers = tuple(self._providers.values())

        candidates: list[SkillDescriptor] = []

        for skill in skills:
            if not skill.enabled:
                continue
            if self._matches(skill, intent, query_value):
                candidates.append(skill)

        for provider in providers:
            try:
                candidates.extend(provider(intent, query))
            except Exception as exc:
                self._emit(
                    "skill_provider_error",
                    error=f"{type(exc).__name__}: {exc}",
                )

        results = self._dedupe_and_rank(candidates, query_value)
        return tuple(results[:limit])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(
        skill: SkillDescriptor,
        intent: IntentResult | None,
        query: str,
    ) -> bool:
        if query:
            haystack = " ".join(
                [
                    skill.name,
                    skill.description,
                    skill.instructions,
                    skill.provider,
                    " ".join(skill.triggers),
                    " ".join(
                        str(value)
                        for value in (skill.metadata.get("actions") or ())
                    ),
                ]
            ).lower()
            return query in haystack

        if intent is None:
            return False

        if intent.intent is IntentType.COMMAND:
            action = str(intent.entities.get("action") or "").strip().lower()
            if action:
                return action in {
                    str(trigger).strip().lower()
                    for trigger in skill.triggers
                } or action in {
                    str(value).strip().lower()
                    for value in (skill.metadata.get("actions") or ())
                }

        normalized = str(intent.normalized_text or "").strip().lower()
        if not normalized:
            return False

        trigger_values = [
            str(trigger).strip().lower()
            for trigger in skill.triggers
            if str(trigger).strip()
        ]
        return any(trigger in normalized for trigger in trigger_values)

    @staticmethod
    def _dedupe_and_rank(
        skills: list[SkillDescriptor],
        query: str,
    ) -> list[SkillDescriptor]:
        seen: set[str] = set()
        unique: list[SkillDescriptor] = []

        for skill in skills:
            if skill.name in seen or not skill.enabled:
                continue
            seen.add(skill.name)
            unique.append(skill)

        unique.sort(
            key=lambda item: (
                -item.priority,
                item.name,
            )
        )
        return unique

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(event, **payload)
