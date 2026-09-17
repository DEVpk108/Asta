from __future__ import annotations

import threading
from collections.abc import Iterable
from time import time
from typing import Any

from .contracts.workspace import WorkspaceState


class WorkspaceManager:
    """Own the current A.S.T.A. workspace state.

    The manager is deliberately LLM-agnostic. External integrations such as
    the HUD, IDE adapters, Git helpers, or future MCP providers can update the
    workspace through this API without coupling those integrations to the
    context builder or model layer.
    """

    def __init__(self, *, event_bus=None):
        self.event_bus = event_bus
        self._state = WorkspaceState()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def state(self) -> WorkspaceState:
        """Return a detached WorkspaceState copy for runtime consumers."""
        with self._lock:
            value = self._state.to_dict()
        return WorkspaceState(**value)

    # ------------------------------------------------------------------
    # Project / repository state
    # ------------------------------------------------------------------

    def update_project(
        self,
        *,
        name: str | None = None,
        path: str | None = None,
        repository: str | None = None,
        branch: str | None = None,
    ) -> WorkspaceState:
        with self._lock:
            if name is not None:
                self._state.project_name = str(name).strip()
            if path is not None:
                self._state.project_path = str(path).strip()
            if repository is not None:
                self._state.repository = str(repository).strip()
            if branch is not None:
                self._state.branch = str(branch).strip()
            return self._commit_locked()

    def set_active_files(self, files: Iterable[str]) -> WorkspaceState:
        return self._set_paths("active_files", files)

    def set_recent_files(self, files: Iterable[str]) -> WorkspaceState:
        return self._set_paths("recent_files", files)

    def add_recent_file(self, path: str) -> WorkspaceState:
        value = str(path).strip()
        if not value:
            return self.state()

        with self._lock:
            items = [value]
            items.extend(item for item in self._state.recent_files if item != value)
            self._state.recent_files = items[:20]
            return self._commit_locked()

    # ------------------------------------------------------------------
    # Runtime environment
    # ------------------------------------------------------------------

    def set_environment(self, values: dict[str, Any]) -> WorkspaceState:
        with self._lock:
            self._state.environment = dict(values)
            return self._commit_locked()

    def set_hardware(self, values: dict[str, Any]) -> WorkspaceState:
        with self._lock:
            self._state.hardware = dict(values)
            return self._commit_locked()

    def set_metadata(self, values: dict[str, Any]) -> WorkspaceState:
        with self._lock:
            self._state.metadata = dict(values)
            return self._commit_locked()

    def update_metadata(self, **values: Any) -> WorkspaceState:
        with self._lock:
            self._state.metadata.update(values)
            return self._commit_locked()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def clear(self) -> WorkspaceState:
        with self._lock:
            self._state = WorkspaceState()
            return self._commit_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_paths(self, field: str, values: Iterable[str]) -> WorkspaceState:
        items: list[str] = []
        for value in values:
            item = str(value).strip()
            if item and item not in items:
                items.append(item)

        with self._lock:
            setattr(self._state, field, items)
            return self._commit_locked()

    def _commit_locked(self) -> WorkspaceState:
        self._state.updated_at = time()
        state = WorkspaceState(**self._state.to_dict())

        if self.event_bus is not None:
            self.event_bus.emit("workspace_updated", workspace=state.to_dict())

        return state
