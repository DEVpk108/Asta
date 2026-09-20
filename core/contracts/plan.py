from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PlanStatus(str, Enum):
    """Lifecycle state of a plan before task execution owns the runtime state."""

    DRAFT = "draft"
    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStepStatus(str, Enum):
    """Execution state of one planned step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass(slots=True)
class PlanStep:
    """One executable intent in a multi-step plan.

    The planner describes what should happen. The executor will later decide
    how to turn the step into a concrete tool request.
    """

    id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    completion_conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.description = str(self.description).strip()
        if not self.id:
            raise ValueError("plan step id must be non-empty")
        if not self.description:
            raise ValueError("plan step description must be non-empty")

        self.depends_on = self._clean_strings(self.depends_on)
        self.required_capabilities = self._clean_strings(
            self.required_capabilities
        )
        self.completion_conditions = self._clean_strings(
            self.completion_conditions
        )
        self.metadata = dict(self.metadata or {})

    @staticmethod
    def _clean_strings(values: list[str] | tuple[str, ...] | None) -> list[str]:
        return [str(value).strip() for value in (values or ()) if str(value).strip()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "required_capabilities": list(self.required_capabilities),
            "completion_conditions": list(self.completion_conditions),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Plan:
    """Structured execution plan for one user goal.

    A Plan is deliberately separate from AgentTask: Plan describes intended
    work; AgentTask will own live execution state, evidence and terminal
    outcome.
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        self.goal = str(self.goal).strip()
        if not self.goal:
            raise ValueError("plan goal must be non-empty")

        self.steps = list(self.steps)
        self.constraints = [
            str(item).strip()
            for item in (self.constraints or ())
            if str(item).strip()
        ]
        self.metadata = dict(self.metadata or {})
        self.validate()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def add_step(self, step: PlanStep) -> None:
        if any(existing.id == step.id for existing in self.steps):
            raise ValueError(f"duplicate plan step id: {step.id}")
        self.steps.append(step)
        self._touch()
        self.validate()

    def validate(self) -> None:
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan contains duplicate step ids")

        known_ids = set(ids)
        for step in self.steps:
            unknown = set(step.depends_on) - known_ids
            if unknown:
                raise ValueError(
                    f"plan step '{step.id}' depends on unknown step(s): "
                    + ", ".join(sorted(unknown))
                )
            if step.id in step.depends_on:
                raise ValueError(
                    f"plan step '{step.id}' cannot depend on itself"
                )

        if self._has_dependency_cycle():
            raise ValueError("plan contains a dependency cycle")

    def _has_dependency_cycle(self) -> bool:
        graph = {step.id: set(step.depends_on) for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False

            visiting.add(node)
            for dependency in graph[node]:
                if visit(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in graph)

    def ready_steps(self) -> list[PlanStep]:
        completed = {
            step.id
            for step in self.steps
            if step.status is PlanStepStatus.COMPLETED
        }

        return [
            step
            for step in self.steps
            if step.status in {PlanStepStatus.PENDING, PlanStepStatus.READY}
            and all(dependency in completed for dependency in step.depends_on)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "constraints": list(self.constraints),
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
