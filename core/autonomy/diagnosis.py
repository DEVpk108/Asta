from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DiagnosisCategory(str, Enum):
    """Bounded failure categories used by autonomous recovery."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TRANSIENT = "transient"
    MISSING_CAPABILITY = "missing_capability"
    INVALID_TARGET = "invalid_target"
    STATE_MISMATCH = "state_mismatch"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class FailureDiagnosis:
    """Structured diagnosis of one failed execution step."""

    category: DiagnosisCategory
    summary: str
    failed_tool: str
    task_id: str | None = None
    step_id: str | None = None
    attempt: int = 1
    confidence: float = 0.0
    recommended_action: str = "replan"
    requires_user: bool = False
    source: str = "fallback"
    model: str | None = None
    latency_ms: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.recommended_action = str(
            self.recommended_action or "replan"
        ).strip().lower()
        self.summary = str(self.summary or "").strip()
        self.failed_tool = str(self.failed_tool or "").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "summary": self.summary,
            "failed_tool": self.failed_tool,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "attempt": int(self.attempt),
            "confidence": float(self.confidence),
            "recommended_action": self.recommended_action,
            "requires_user": bool(self.requires_user),
            "source": self.source,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }


class DiagnosisEngine:
    """Provider-agnostic failure diagnosis boundary.

    The engine delegates bounded classification to the configured System-1
    provider when that provider exposes diagnose_failure. It always keeps a
    deterministic fallback so diagnosis remains available when Laya is disabled
    or unavailable.
    """

    def __init__(self, decision_engine=None):
        self.decision_engine = decision_engine

    def diagnose(
        self,
        task,
        result,
        *,
        recovery=None,
        step_id: str | None = None,
    ) -> FailureDiagnosis:
        payload = self._state(task, result, recovery=recovery, step_id=step_id)
        provider = getattr(self.decision_engine, "diagnose_failure", None)

        if callable(provider):
            try:
                diagnosis = provider(payload)
            except Exception as exc:
                fallback = self._fallback(payload)
                fallback.metadata["provider_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return fallback
            if isinstance(diagnosis, FailureDiagnosis):
                return diagnosis

        return self._fallback(payload)

    @staticmethod
    def _state(task, result, *, recovery=None, step_id=None) -> dict[str, Any]:
        evidence = getattr(task, "evidence", ()) or ()
        recent = []
        for item in list(evidence)[-6:]:
            if not isinstance(item, dict):
                continue
            recent.append(
                {
                    "tool": item.get("tool"),
                    "success": item.get("success"),
                    "error": item.get("error"),
                    "plan_step_id": item.get("plan_step_id"),
                }
            )

        output = getattr(result, "output", None)
        if isinstance(output, (str, int, float, bool)) or output is None:
            output_text = str(output) if output is not None else ""
        else:
            output_text = repr(output)

        recovery_payload = (
            recovery.to_dict()
            if hasattr(recovery, "to_dict")
            else dict(recovery or {})
        )

        return {
            "goal": str(getattr(task, "goal", "") or ""),
            "failed_tool": str(getattr(result, "tool", "") or ""),
            "step_id": step_id,
            "attempt": recovery_payload.get("attempt", 1),
            "error": str(getattr(result, "error", "") or "")[:1200],
            "output": output_text[:1200],
            "recovery": recovery_payload,
            "recent_evidence": recent,
        }

    @staticmethod
    def _fallback(payload: dict[str, Any]) -> FailureDiagnosis:
        error = str(payload.get("error") or "").lower()
        tool = str(payload.get("failed_tool") or "")
        category = DiagnosisCategory.UNKNOWN
        recommended = "replan"
        requires_user = False

        if any(
            marker in error
            for marker in (
                "authentication required",
                "login required",
                "sign in required",
                "oauth",
                "token expired",
                "credentials",
            )
        ):
            category = DiagnosisCategory.AUTHENTICATION
            recommended = "wait_for_user"
            requires_user = True
        elif any(
            marker in error
            for marker in (
                "authorization required",
                "forbidden",
                "permission denied",
                "access denied",
            )
        ):
            category = DiagnosisCategory.AUTHORIZATION
            recommended = "wait_for_user"
            requires_user = True
        elif any(
            marker in error
            for marker in (
                "timeout",
                "temporarily unavailable",
                "temporary failure",
                "connection reset",
                "connection refused",
                "network error",
                "busy",
            )
        ):
            category = DiagnosisCategory.TRANSIENT
            recommended = "retry"
        elif any(
            marker in error
            for marker in (
                "tool not found",
                "unknown tool",
                "unsupported capability",
                "not registered",
            )
        ):
            category = DiagnosisCategory.MISSING_CAPABILITY
        elif any(
            marker in error
            for marker in (
                "invalid target",
                "application not found",
                "window not found",
                "track not found",
                "file not found",
            )
        ):
            category = DiagnosisCategory.INVALID_TARGET
        elif any(
            marker in error
            for marker in (
                "not running",
                "no active device",
                "invalid state",
                "state mismatch",
                "not open",
            )
        ):
            category = DiagnosisCategory.STATE_MISMATCH

        summary = (
            f"Failure in {tool or 'tool execution'} classified as "
            f"{category.value}."
        )
        return FailureDiagnosis(
            category=category,
            summary=summary,
            failed_tool=tool,
            step_id=payload.get("step_id"),
            attempt=int(payload.get("attempt") or 1),
            confidence=(
                0.55
                if category is not DiagnosisCategory.UNKNOWN
                else 0.30
            ),
            recommended_action=recommended,
            requires_user=requires_user,
            source="fallback",
            evidence={
                "error": str(payload.get("error") or "")[:1200],
                "output": str(payload.get("output") or "")[:1200],
            },
        )
