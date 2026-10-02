from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AgentBelief:
    key: str
    value: Any
    confidence: float = 0.5
    source: str = "inference"
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.key = str(self.key).strip()
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.source = str(self.source or "inference").strip() or "inference"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentObservation:
    kind: str
    summary: str
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip()
        self.summary = str(self.summary).strip()
        self.source = str(self.source).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentState:
    goal: str
    success_conditions: list[str] = field(default_factory=list)
    beliefs: dict[str, AgentBelief] = field(default_factory=dict)
    observations: list[AgentObservation] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    current_strategy: str = ""
    uncertainty: float = 1.0

    def __post_init__(self) -> None:
        self.goal = str(self.goal).strip()
        self.success_conditions = [
            str(item).strip()
            for item in self.success_conditions
            if str(item).strip()
        ]
        self.current_strategy = str(self.current_strategy or "").strip()
        self.uncertainty = max(0.0, min(1.0, float(self.uncertainty)))

    def set_belief(
        self,
        key: str,
        value: Any,
        *,
        confidence: float,
        source: str,
    ) -> None:
        belief = AgentBelief(
            key=key,
            value=value,
            confidence=confidence,
            source=source,
        )
        self.beliefs[belief.key] = belief
        self._refresh_uncertainty()

    def observe(
        self,
        kind: str,
        summary: str,
        *,
        source: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.observations.append(
            AgentObservation(
                kind=kind,
                summary=summary,
                source=source,
                data=dict(data or {}),
            )
        )
        if len(self.observations) > 32:
            del self.observations[:-32]

    def record_action(
        self,
        *,
        action: str,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        outcome: str | None = None,
    ) -> None:
        self.actions.append(
            {
                "action": str(action).strip(),
                "tool": str(tool).strip() if tool else None,
                "arguments": dict(arguments or {}),
                "outcome": str(outcome).strip() if outcome else None,
                "timestamp": _now(),
            }
        )
        if len(self.actions) > 32:
            del self.actions[:-32]

    def _refresh_uncertainty(self) -> None:
        if not self.beliefs:
            self.uncertainty = 1.0
            return
        average_confidence = sum(
            belief.confidence for belief in self.beliefs.values()
        ) / len(self.beliefs)
        self.uncertainty = max(0.0, min(1.0, 1.0 - average_confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success_conditions": list(self.success_conditions),
            "beliefs": {
                key: belief.to_dict()
                for key, belief in self.beliefs.items()
            },
            "observations": [
                observation.to_dict()
                for observation in self.observations
            ],
            "actions": [dict(action) for action in self.actions],
            "current_strategy": self.current_strategy,
            "uncertainty": self.uncertainty,
        }
