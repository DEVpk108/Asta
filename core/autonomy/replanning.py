from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReplanStrategy(str, Enum):
    """Bounded plan-repair strategies available to the task runtime."""

    WAIT_FOR_USER = "wait_for_user"
    RESTORE_STATE = "restore_state"
    REBUILD_PLAN = "rebuild_plan"
    FAIL = "fail"


@dataclass(slots=True)
class ReplanDecision:
    strategy: ReplanStrategy
    reason: str
    confidence: float = 0.0
    source: str = "fallback"
    model: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.reason = str(self.reason or "").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "source": self.source,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


class ReplanEngine:
    """Choose a bounded repair strategy without executing it.

    Laya may select among the finite strategies. Deterministic guards remain
    authoritative for user-only security boundaries and unsupported repairs.
    """

    def __init__(self, decision_engine=None):
        self.decision_engine = decision_engine

    def choose(
        self,
        diagnosis,
        *,
        plan_step=None,
    ) -> ReplanDecision:
        if getattr(diagnosis, "requires_user", False):
            return ReplanDecision(
                strategy=ReplanStrategy.WAIT_FOR_USER,
                reason="The diagnosis requires a user-only security or confirmation step.",
                confidence=1.0,
                source="guard",
            )

        provider = getattr(
            self.decision_engine,
            "select_replan_strategy",
            None,
        )
        if callable(provider):
            try:
                result = provider(diagnosis, plan_step=plan_step)
            except Exception as exc:
                fallback = self._fallback(diagnosis, plan_step=plan_step)
                fallback.metadata["provider_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return fallback
            if isinstance(result, ReplanDecision):
                if (
                    result.confidence >= 0.55
                    or result.strategy is ReplanStrategy.WAIT_FOR_USER
                ):
                    return result
                fallback = self._fallback(diagnosis, plan_step=plan_step)
                fallback.metadata["low_confidence_provider"] = True
                fallback.metadata["provider_confidence"] = result.confidence
                return fallback

        return self._fallback(diagnosis, plan_step=plan_step)

    @staticmethod
    def _fallback(diagnosis, *, plan_step=None) -> ReplanDecision:
        category = getattr(diagnosis, "category", None)
        value = getattr(category, "value", str(category or "unknown"))

        if value in {"authentication", "authorization", "setup_required"}:
            return ReplanDecision(
                strategy=ReplanStrategy.WAIT_FOR_USER,
                reason="The task cannot proceed until the user completes an access boundary.",
                confidence=1.0,
                source="guard",
            )

        if value == "state_mismatch":
            return ReplanDecision(
                strategy=ReplanStrategy.RESTORE_STATE,
                reason="The environment state does not satisfy the step precondition; restore the target state and retry.",
                confidence=0.75,
            )

        if value in {
            "transient",
            "invalid_target",
            "missing_capability",
            "unknown",
        }:
            return ReplanDecision(
                strategy=ReplanStrategy.REBUILD_PLAN,
                reason="The current execution assumptions should be rebuilt before continuing.",
                confidence=0.60,
            )

        return ReplanDecision(
            strategy=ReplanStrategy.FAIL,
            reason="No safe bounded repair strategy is available.",
            confidence=0.80,
        )
