from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """Structured System-1 decision output for one user request."""

    engine: str
    model: str | None
    input_text: str
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "input_text": self.input_text,
            "decisions": {
                key: dict(value)
                for key, value in self.decisions.items()
            },
            "routing": dict(self.routing),
            "latency_ms": self.latency_ms,
        }

    def choice(self, question_id: str) -> str | None:
        answer = self.decisions.get(str(question_id))
        if not answer:
            return None
        value = answer.get("choice")
        return str(value) if value is not None else None

    def probability(self, question_id: str, key: str = "noul") -> float | None:
        answer = self.decisions.get(str(question_id))
        if not answer:
            return None
        value = answer.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def score(self, question_id: str) -> float | None:
        answer = self.decisions.get(str(question_id))
        if not answer:
            return None
        value = answer.get("score")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
