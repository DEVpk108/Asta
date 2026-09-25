from __future__ import annotations

import threading
from uuid import uuid4
from typing import Any, Iterable

from .contracts.plan import Plan
from .contracts.plan import PlanStepStatus
from .contracts.task import AgentTask, TaskStatus


class TaskManager:
    """Own the active task and the in-memory task history for A.S.T.A.

    The manager is intentionally LLM-agnostic. It provides a small runtime
    API that later layers such as workspace state, capability discovery,
    memory, MCP and reflection can build around.
    """

    def __init__(self, *, event_bus=None):
        self.event_bus = event_bus
        self._tasks: dict[str, AgentTask] = {}
        self._active_task_id: str | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Task creation / lookup
    # ------------------------------------------------------------------

    def create(
        self,
        goal: str,
        *,
        constraints: Iterable[str] | None = None,
        pending_steps: Iterable[str] | None = None,
        plan: Plan | None = None,
        metadata: dict[str, Any] | None = None,
        activate: bool = True,
    ) -> AgentTask:
        value = str(goal).strip()
        if not value:
            raise ValueError("task goal must be non-empty")

        task = AgentTask(
            id=str(uuid4()),
            goal=value,
            constraints=[str(item).strip() for item in (constraints or ()) if str(item).strip()],
            pending_steps=[str(item).strip() for item in (pending_steps or ()) if str(item).strip()],
            plan=plan,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._tasks[task.id] = task

            if activate:
                self._activate_locked(task)

            snapshot = task.to_dict()

        self._emit("task_created", task=snapshot)
        if activate:
            self._emit("task_activated", task=snapshot)
        return task

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def current(self) -> AgentTask | None:
        with self._lock:
            if self._active_task_id is None:
                return None
            return self._tasks.get(self._active_task_id)

    def snapshot(self, task_id: str | None = None) -> dict[str, Any] | None:
        task = self.current() if task_id is None else self.get(task_id)
        return task.to_dict() if task is not None else None

    def list(self, *, status: TaskStatus | None = None) -> list[AgentTask]:
        with self._lock:
            tasks = list(self._tasks.values())
            if status is not None:
                tasks = [task for task in tasks if task.status == status]
            return list(tasks)

    # ------------------------------------------------------------------
    # Active task management
    # ------------------------------------------------------------------

    def activate(self, task_id: str) -> AgentTask:
        with self._lock:
            task = self._require(task_id)
            self._activate_locked(task)
            snapshot = task.to_dict()

        self._emit("task_activated", task=snapshot)
        return task

    def pause(self, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.pause()
            if self._active_task_id == task.id:
                self._active_task_id = None
            snapshot = task.to_dict()

        self._emit("task_paused", task=snapshot)
        return task

    def resume(self, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            self._activate_locked(task)
            if task.status != TaskStatus.ACTIVE:
                task.resume()
            snapshot = task.to_dict()

        self._emit("task_resumed", task=snapshot)
        return task

    # ------------------------------------------------------------------
    # Task progress
    # ------------------------------------------------------------------

    def set_step(self, step: str, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.set_current_step(step)
            snapshot = task.to_dict()

        self._emit("task_updated", task=snapshot)
        return task

    def complete_step(self, step: str, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.complete_step(step)
            snapshot = task.to_dict()

        self._emit("task_updated", task=snapshot)
        return task

    def set_plan_step_status(
        self,
        step_id: str,
        status: PlanStepStatus,
        task_id: str | None = None,
    ) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            if task.plan is None:
                raise ValueError("task has no structured plan")
            task.plan.set_step_status(step_id, status)
            snapshot = task.to_dict()

        self._emit("task_updated", task=snapshot)
        return task

    def refresh_ready_plan_steps(self, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            if task.plan is None:
                raise ValueError("task has no structured plan")
            task.plan.mark_ready_steps()
            snapshot = task.to_dict()

        self._emit("task_updated", task=snapshot)
        return task

    def add_evidence(self, evidence: dict[str, Any] | Any, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.add_evidence(evidence)
            snapshot = task.to_dict()

        self._emit("task_updated", task=snapshot)
        return task

    # ------------------------------------------------------------------
    # Terminal states
    # ------------------------------------------------------------------

    def complete(self, result: Any = None, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.complete(result=result)
            if self._active_task_id == task.id:
                self._active_task_id = None
            snapshot = task.to_dict()

        self._emit("task_completed", task=snapshot)
        return task

    def fail(self, error: str, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.fail(error)
            if self._active_task_id == task.id:
                self._active_task_id = None
            snapshot = task.to_dict()

        self._emit("task_failed", task=snapshot)
        return task

    def cancel(self, reason: str | None = None, task_id: str | None = None) -> AgentTask:
        with self._lock:
            task = self._require(task_id or self._active_task_id)
            task.cancel(reason)
            if self._active_task_id == task.id:
                self._active_task_id = None
            snapshot = task.to_dict()

        self._emit("task_cancelled", task=snapshot)
        return task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activate_locked(self, task: AgentTask) -> None:
        previous_id = self._active_task_id
        if previous_id and previous_id != task.id:
            previous = self._tasks.get(previous_id)
            if previous and previous.status == TaskStatus.ACTIVE:
                previous.pause()
                self._emit("task_paused", task=previous.to_dict())

        if task.status == TaskStatus.PAUSED:
            task.resume()
        else:
            task.activate()
        self._active_task_id = task.id

    def _require(self, task_id: str | None) -> AgentTask:
        if not task_id:
            raise ValueError("no active task")
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        return task

    def _emit(self, event: str, **payload: Any) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(event, **payload)
