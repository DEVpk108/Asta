from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecoveryAction(str, Enum):
    """What the task runtime should do after an execution failure."""

    RETRY = "retry"
    REPLAN = "replan"
    WAIT_FOR_USER = "wait_for_user"
    FAIL = "fail"


@dataclass(slots=True)
class RecoveryDecision:
    """Structured recovery decision produced from one failed tool step."""

    action: RecoveryAction
    reason: str
    failed_tool: str
    task_id: str | None = None
    step_id: str | None = None
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "failed_tool": self.failed_tool,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "metadata": dict(self.metadata),
        }


class RecoveryManager:
    """Decide what happens after a tool failure.

    This is intentionally separate from the LLM planner. The first version
    only establishes a deterministic recovery contract and retry budget.
    Later versions can plug diagnosis and LLM-backed replanning into the
    same interface without changing TaskRuntime's failure boundary.
    """

    def __init__(self, *, max_retries: int = 1):
        self.max_retries = max(0, int(max_retries))

    def decide(self, task, result, *, step_id: str | None = None) -> RecoveryDecision:
        """Return a recovery action without executing anything."""

        attempts = self._attempt_count(task, result, step_id)
        task_id = getattr(task, "id", None)

        if self._needs_user(result):
            return RecoveryDecision(
                action=RecoveryAction.WAIT_FOR_USER,
                reason=self._user_reason(result),
                failed_tool=str(result.tool),
                task_id=task_id,
                step_id=step_id,
                attempt=attempts,
                metadata={"requires_user": True},
            )

        if attempts <= self.max_retries:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                reason="The failed step has not exhausted its retry budget.",
                failed_tool=str(result.tool),
                task_id=task_id,
                step_id=step_id,
                attempt=attempts,
            )

        return RecoveryDecision(
            action=RecoveryAction.REPLAN,
            reason=(
                "The step failed after the retry budget was exhausted. "
                "The environment or the original assumptions may be wrong; "
                "the task should be diagnosed and replanned."
            ),
            failed_tool=str(result.tool),
            task_id=task_id,
            step_id=step_id,
            attempt=attempts,
            metadata={"retry_budget_exhausted": True},
        )

    @staticmethod
    def _attempt_count(task, result, step_id: str | None) -> int:
        evidence = getattr(task, "evidence", ()) or ()
        count = 0
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_result" or item.get("success") is not False:
                continue
            if item.get("tool") != result.tool:
                continue
            if step_id and item.get("plan_step_id") not in {None, step_id}:
                continue
            count += 1
        return count

    @staticmethod
    def _needs_user(result) -> bool:
        metadata = getattr(result, "metadata", {}) or {}
        if metadata.get("requires_confirmation"):
            return True

        error = str(getattr(result, "error", "") or "").lower()
        return any(
            marker in error
            for marker in (
                "authentication required",
                "authorization required",
                "login required",
                "requires user",
                "user confirmation",
            )
        )

    @staticmethod
    def _user_reason(result) -> str:
        error = str(getattr(result, "error", "") or "").strip()
        if error:
            return error
        return "The task requires user authentication or confirmation."
