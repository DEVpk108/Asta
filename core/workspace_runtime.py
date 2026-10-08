from __future__ import annotations

import os
import platform
import subprocess
import sys
import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .module import Module
from .workspace_store import WorkspaceStateStore


logger = logging.getLogger(__name__)


class WorkspaceRuntimeModule(Module):
    """Populate WorkspaceManager with runtime-discoverable workspace state."""

    def __init__(self, kernel, *, state_path=None):
        super().__init__(
            name="WorkspaceRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.state_store = WorkspaceStateStore(state_path)

    def initialize(self):
        self._restore_workspace()
        self.event_bus.subscribe("workspace_updated", self.on_workspace_updated)
        self.event_bus.subscribe("workspace_refresh", self.on_refresh)
        self.refresh()
        print("[Workspace] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("workspace_updated", self.on_workspace_updated)
        self.event_bus.unsubscribe("workspace_refresh", self.on_refresh)
        print("[Workspace] Stopped", flush=True)

    def _restore_workspace(self):
        state = self.state_store.load()
        manager = self.kernel.workspace_manager
        current = manager.snapshot()
        if not state or current.get("project_path"):
            return
        manager.update_project(
            name=state.get("project_name"),
            path=state.get("project_path"),
            repository=state.get("repository"),
            branch=state.get("branch"),
        )
        manager.set_active_files(state.get("active_files", []))
        manager.set_recent_files(state.get("recent_files", []))

    def on_workspace_updated(self, workspace=None, **kwargs):
        if not isinstance(workspace, dict):
            return
        try:
            self.state_store.save(workspace)
        except (OSError, ValueError, TypeError):
            logger.exception("Could not persist the workspace context.")

    def on_refresh(self):
        self.refresh()

    def refresh(self):
        manager = self.kernel.workspace_manager
        previous = manager.snapshot()
        root = self._workspace_root(previous.get("project_path", ""))
        previous_path = str(previous.get("project_path") or "").strip()
        try:
            project_changed = bool(previous_path) and (
                Path(previous_path).expanduser().resolve() != root
            )
        except (OSError, RuntimeError):
            project_changed = bool(previous_path)
        repository = self._git_value(
            root,
            ["git", "config", "--get", "remote.origin.url"],
        ) or ("" if project_changed else str(previous.get("repository") or ""))
        branch = self._git_value(
            root,
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        ) or ("" if project_changed else str(previous.get("branch") or ""))

        manager.update_project(
            name=root.name,
            path=str(root),
            repository=self._normalize_repository(repository),
            branch=branch,
            reset_file_history=project_changed,
        )
        manager.set_environment(
            {
                "os": platform.system(),
                "os_release": platform.release(),
                "python": platform.python_version(),
                "python_executable": sys.executable,
                "cwd": str(Path.cwd()),
            }
        )
        manager.set_hardware(
            {
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
        )

    @staticmethod
    def _workspace_root(persisted_path: str = "") -> Path:
        configured = os.getenv("ASTA_WORKSPACE_PATH", "").strip()
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.is_dir():
                return path

        if persisted_path:
            try:
                path = Path(persisted_path).expanduser().resolve()
                if path.is_dir():
                    return path
            except (OSError, RuntimeError):
                pass

        root = WorkspaceRuntimeModule._git_toplevel(Path.cwd())
        if root is not None:
            return root

        return Path.cwd().resolve()

    @staticmethod
    def _git_toplevel(path: Path) -> Path | None:
        value = WorkspaceRuntimeModule._git_value(
            path,
            ["git", "rev-parse", "--show-toplevel"],
        )
        if not value:
            return None

        candidate = Path(value).expanduser().resolve()
        return candidate if candidate.exists() else None

    @staticmethod
    def _git_value(cwd: Path, command: list[str]) -> str:
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""

        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    @staticmethod
    def _normalize_repository(value: str) -> str:
        if not value:
            return ""

        # Avoid preserving credentials that may appear in an HTTPS remote.
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            hostname = parsed.hostname or parsed.netloc
            netloc = hostname
            try:
                if parsed.port:
                    netloc = f"{hostname}:{parsed.port}"
            except ValueError:
                return ""
            return urlunsplit(
                (parsed.scheme, netloc, parsed.path, "", "")
            )

        if "@" in value:
            host, separator, repository_path = value.partition(":")
            if separator:
                return f"{host.rsplit('@', 1)[-1]}:{repository_path}"
        return value
