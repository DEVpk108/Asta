from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum


class CapabilitySetupStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CapabilitySetupEvent:
    capability: str
    task_id: str
    step_id: str | None
    status: CapabilitySetupStatus
    summary: str
    requires_user: bool = False
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "status": self.status.value,
            "summary": self.summary,
            "requires_user": self.requires_user,
            "metadata": dict(self.metadata or {}),
        }


class CapabilitySetupManager:
    """Run bounded environment-repair operators outside the task event thread."""

    def __init__(self, media_manager, *, event_bus=None, operators=None):
        self.media_manager = media_manager
        self.event_bus = event_bus
        self._operators = dict(operators or {"spotify": self._spotify_operator})
        self._running: set[tuple[str, str]] = set()
        self._lock = threading.RLock()

    def supports(self, capability: str) -> bool:
        return str(capability or "").strip().lower() in self._operators

    def start(self, task, *, capability: str, step_id: str | None = None) -> bool:
        key = (task.id, str(capability or "").strip().lower())
        if not self.supports(key[1]):
            return False

        with self._lock:
            if key in self._running:
                return True
            self._running.add(key)

        event = CapabilitySetupEvent(
            capability=key[1],
            task_id=task.id,
            step_id=step_id,
            status=CapabilitySetupStatus.STARTED,
            summary=f"Starting capability setup for {key[1]}.",
        )
        print(
            f"[Setup] Starting capability setup: {key[1]} "
            f"(task={task.id}, step={step_id or 'unknown'})",
            flush=True,
        )
        self._emit("capability_setup_started", event)

        thread = threading.Thread(
            target=self._run,
            args=(task.id, key[1], step_id),
            name=f"asta-setup-{key[1]}",
            daemon=True,
        )
        thread.start()
        return True

    def _run(self, task_id: str, capability: str, step_id: str | None) -> None:
        key = (task_id, capability)
        try:
            operator = self._operators[capability]
            result = operator()
            status = getattr(result, "status", None)
            status_value = getattr(status, "value", str(status)).lower()

            if status_value == "completed":
                self.media_manager.refresh_configuration()
                event = CapabilitySetupEvent(
                    capability=capability,
                    task_id=task_id,
                    step_id=step_id,
                    status=CapabilitySetupStatus.COMPLETED,
                    summary=str(getattr(result, "summary", "Setup completed.")),
                    metadata={
                        "redirect_uri": getattr(result, "redirect_uri", None),
                    },
                )
            elif status_value == "waiting_for_user":
                event = CapabilitySetupEvent(
                    capability=capability,
                    task_id=task_id,
                    step_id=step_id,
                    status=CapabilitySetupStatus.WAITING_FOR_USER,
                    summary=str(getattr(result, "summary", "User action is required.")),
                    requires_user=True,
                    metadata=dict(getattr(result, "metadata", None) or {}),
                )
            else:
                event = CapabilitySetupEvent(
                    capability=capability,
                    task_id=task_id,
                    step_id=step_id,
                    status=CapabilitySetupStatus.FAILED,
                    summary=str(getattr(result, "summary", "Capability setup failed.")),
                    metadata=dict(getattr(result, "metadata", None) or {}),
                )
        except Exception as exc:
            event = CapabilitySetupEvent(
                capability=capability,
                task_id=task_id,
                step_id=step_id,
                status=CapabilitySetupStatus.FAILED,
                summary=f"Capability setup failed: {type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                self._running.discard(key)

        print(
            f"[Setup] {capability} setup {event.status.value}: {event.summary}",
            flush=True,
        )
        if event.status in {
            CapabilitySetupStatus.WAITING_FOR_USER,
            CapabilitySetupStatus.FAILED,
        }:
            self._notify(event.summary)
        self._emit("capability_setup_completed", event)

    def _spotify_operator(self):
        from .spotify_setup import SpotifySetupOperator

        return SpotifySetupOperator(
            notify=self._notify,
        ).run()

    def _notify(self, message: str) -> None:
        if self.event_bus is None:
            return
        self.event_bus.emit("assistant_sentence", text=str(message))
        self.event_bus.emit("assistant_response", text=str(message))

    def _emit(self, event_name: str, event: CapabilitySetupEvent) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(
                event_name,
                event=event.to_dict(),
            )
