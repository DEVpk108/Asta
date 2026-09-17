from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .module import Module


class WorkspaceRuntimeModule(Module):
    """Populate WorkspaceManager with runtime-discoverable workspace state."""

    def __init__(self, kernel):
        super().__init__(
            name="WorkspaceRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

    def initialize(self):
        self.event_bus.subscribe("workspace_refresh", self.on_refresh)
        self.refresh()
        print("[Workspace] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("workspace_refresh", self.on_refresh)
        print("[Workspace] Stopped", flush=True)

    def on_refresh(self):
        self.refresh()

    def refresh(self):
        manager = self.kernel.workspace_manager
        root = self._workspace_root()

        repository = self._git_value(
            root,
            ["git", "config", "--get", "remote.origin.url"],
        )
        branch = self._git_value(
            root,
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        )

        manager.update_project(
            name=root.name,
            path=str(root),
            repository=self._normalize_repository(repository),
            branch=branch,
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

    @staticmethod
    def _workspace_root() -> Path:
        configured = os.getenv("ASTA_WORKSPACE_PATH", "").strip()
        if configured:
            path = Path(configured).expanduser().resolve()
            if path.exists():
                return path

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
            if parsed.port:
                netloc = f"{hostname}:{parsed.port}"
            return urlunsplit(
                (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
            )

        return value
