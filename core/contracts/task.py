from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .plan import Plan


class TaskStatus(str, Enum):
    """Lifecycle states for an A.S.T.A. agent task."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class AgentTask:
    """Mutable state describing one user goal across multiple agent steps.

    The task is deliberately independent from the LLM and from tools. It is
    the runtime's durable-in-memory representation of what A.S.T.A. is trying
    to accomplish, what it has already done, and what remains.
    """

    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING

    constraints: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    current_step: str | None = None
    plan: Plan | None = None

    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    result: Any = None
    error: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def activate(self) -> None:
        if self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            raise ValueError(f"cannot activate task in {self.status.value} state")
        self.status = TaskStatus.ACTIVE
        if self.plan is not None:
            self.plan.status = self.plan.status.ACTIVE
            self.plan.mark_ready_steps()
        self._touch()

    def pause(self) -> None:
        if self.status != TaskStatus.ACTIVE:
            raise ValueError(f"cannot pause task in {self.status.value} state")
        self.status = TaskStatus.PAUSED
        self._touch()

    def resume(self) -> None:
        if self.status != TaskStatus.PAUSED:
            raise ValueError(f"cannot resume task in {self.status.value} state")
        self.status = TaskStatus.ACTIVE
        self._touch()

    def complete(self, result: Any = None) -> None:
        if self.status not in {TaskStatus.ACTIVE, TaskStatus.PAUSED, TaskStatus.PENDING}:
            raise ValueError(f"cannot complete task in {self.status.value} state")
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.current_step = None
        self.error = None
        if self.plan is not None:
            self.plan.status = self.plan.status.COMPLETED
            for step in self.plan.steps:
                if step.status not in {
                    step.status.COMPLETED,
                    step.status.SKIPPED,
                }:
                    step.status = step.status.SKIPPED
            self.plan.updated_at = datetime.now(timezone.utc)
        self._touch()

    def fail(self, error: str) -> None:
        if self.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
            raise ValueError(f"cannot fail task in {self.status.value} state")
        self.status = TaskStatus.FAILED
        self.error = str(error)
        self.current_step = None
        if self.plan is not None:
            self.plan.status = self.plan.status.FAILED
            self.plan.updated_at = datetime.now(timezone.utc)
        self._touch()

    def cancel(self, reason: str | None = None) -> None:
        if self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            raise ValueError(f"cannot cancel task in {self.status.value} state")
        self.status = TaskStatus.CANCELLED
        self.error = str(reason) if reason else self.error
        self.current_step = None
        if self.plan is not None:
            self.plan.status = self.plan.status.CANCELLED
            self.plan.updated_at = datetime.now(timezone.utc)
        self._touch()

    def set_current_step(self, step: str | None) -> None:
        self.current_step = step.strip() if step else None
        self._touch()

    def add_pending_step(self, step: str) -> None:
        value = step.strip()
        if value and value not in self.pending_steps:
            self.pending_steps.append(value)
            self._touch()

    def complete_step(self, step: str) -> None:
        value = step.strip()
        if not value:
            return
        if value in self.pending_steps:
            self.pending_steps.remove(value)
        if value not in self.completed_steps:
            self.completed_steps.append(value)
        if self.current_step == value:
            self.current_step = None
        self._touch()

    def add_evidence(self, evidence: dict[str, Any] | Any) -> None:
        if isinstance(evidence, dict):
            self.evidence.append(dict(evidence))
        else:
            self.evidence.append({"value": evidence})
        self._touch()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the current task state."""
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "constraints": list(self.constraints),
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "current_step": self.current_step,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "evidence": [dict(item) for item in self.evidence],
            "metadata": dict(self.metadata),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
