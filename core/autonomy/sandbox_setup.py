"""Self-repair for the code sandbox (Docker/Podman).

When a project command can't run because the container engine is stopped or
the sandbox image was never built, ASTA should not just say "start Docker".
This operator *observes* the environment (is the CLI installed? is the engine
answering? is the image present?), applies the matching fix, checks that the
fix worked, and reports what it saw and did.

Every action is a fixed, vetted recipe (start Docker Desktop, `podman machine
start`, build ASTA's own sandbox/Dockerfile). Nothing here runs model-written
commands on the host.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.tools.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    SandboxRuntimeError,
    _bounded_process,
    _resolve_runtime_executable,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class SandboxSetupStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class SandboxSetupResult:
    status: SandboxSetupStatus
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _env_seconds(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


DESKTOP_EXE = "Docker Desktop.exe"


def _registry_install_locations() -> list[Path]:
    """Docker Desktop's InstallLocation from the Windows uninstall registry."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return []
    found = []
    key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Docker Desktop"
    for hive in (getattr(winreg, "HKEY_LOCAL_MACHINE", None), getattr(winreg, "HKEY_CURRENT_USER", None)):
        if hive is None:
            continue
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "InstallLocation")
        except OSError:
            continue
        if value:
            found.append(Path(str(value)) / DESKTOP_EXE)
    return found


def _docker_desktop_candidates(cli_path: str | None = None) -> list[Path]:
    """Where Docker Desktop.exe may be, most specific first."""
    configured = os.getenv("ASTA_DOCKER_DESKTOP_PATH", "").strip()
    paths = [Path(configured)] if configured else []
    if cli_path:
        # docker.exe ships inside Docker Desktop (...\Docker\resources\bin),
        # so walking up from the CLI finds custom install folders too.
        for parent in list(Path(cli_path).resolve().parents)[:5]:
            paths.append(parent / DESKTOP_EXE)
    paths.extend(_registry_install_locations())
    for base in (os.getenv("ProgramFiles"), os.getenv("ProgramW6432"), os.getenv("LOCALAPPDATA")):
        if base:
            paths.append(Path(base) / "Docker" / "Docker" / DESKTOP_EXE)
            paths.append(Path(base) / "Programs" / "Docker" / "Docker" / DESKTOP_EXE)
    unique, seen = [], set()
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


class SandboxSetupOperator:
    """Diagnose -> fix -> verify for the container sandbox."""

    def __init__(
        self,
        *,
        notify: Callable[[str], None] | None = None,
        runtime_name: str | None = None,
        image: str | None = None,
        resolve: Callable[[str], str | None] = _resolve_runtime_executable,
        run: Callable[..., Any] = _bounded_process,
        launch: Callable[[list[str]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        platform: str | None = None,
        repo_root: Path | None = None,
        application_manager=None,
        open_target: Callable[[str], None] | None = None,
    ):
        self.notify = notify or (lambda message: None)
        self.runtime_name = (runtime_name or os.getenv("ASTA_CONTAINER_RUNTIME", "docker")).strip().lower()
        self.image = (image or os.getenv("ASTA_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)).strip()
        self.resolve = resolve
        self._run = run
        self._launch = launch or self._detached_launch
        self.sleep = sleep
        self.clock = clock
        self.platform = platform or sys.platform
        self.repo_root = Path(repo_root or REPO_ROOT)
        self.application_manager = application_manager
        self._open_target = open_target or self._shell_open
        self.startup_timeout = _env_seconds("ASTA_SANDBOX_STARTUP_TIMEOUT", 180.0)
        self.build_timeout = _env_seconds("ASTA_SANDBOX_BUILD_TIMEOUT", 1200.0)
        self.auto_build = os.getenv("ASTA_SANDBOX_AUTO_BUILD", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self.observations: list[str] = []
        self.actions: list[str] = []

    # ------------------------------------------------------------------
    def run(self, context: dict[str, Any] | None = None) -> SandboxSetupResult:
        executable = self.resolve(self.runtime_name)
        if not executable:
            self.observations.append(f"{self.runtime_name} CLI not found")
            return self._failed(
                f"I can't create the code sandbox because {self._product} isn't installed "
                "(or isn't on PATH). Install it, then ask me again.",
                fixable=False,
            )
        self.executable = executable

        if self._engine_ready():
            self.observations.append(f"{self.runtime_name} engine is running")
        else:
            self.observations.append(f"{self.runtime_name} engine is not running")
            started = self._start_engine()
            if not started:
                return self._failed(
                    f"{self._product} isn't running and I couldn't find it to start it "
                    "(I tried the docker desktop command, the folder of the docker CLI, the "
                    "installed-programs list and the Start menu). Start it once yourself, or set "
                    "ASTA_DOCKER_DESKTOP_PATH to Docker Desktop.exe, then ask me again.",
                    fixable=False,
                )
            self.notify(
                f"The code sandbox needs {self._product}, which isn't running. "
                "I'm starting it now; this can take a minute."
            )
            if not self._wait_for_engine():
                return self._failed(
                    f"I started {self._product}, but it wasn't ready after "
                    f"{int(self.startup_timeout)} seconds. Check that it finished starting, "
                    "then ask me again."
                )
            self.observations.append(f"{self.runtime_name} engine answered after start")

        if self._image_present():
            self.observations.append(f"sandbox image {self.image} is present")
        else:
            self.observations.append(f"sandbox image {self.image} is missing")
            dockerfile = self.repo_root / "sandbox" / "Dockerfile"
            if not self.auto_build or not dockerfile.is_file():
                return self._failed(
                    "The sandbox image isn't built. In the ASTA folder run "
                    f"`{self.runtime_name} build --pull -t {self.image} -f sandbox/Dockerfile .`, "
                    "then ask me again."
                )
            self.notify(
                "The sandbox image isn't built yet. I'm building it once from ASTA's own "
                "Dockerfile; this needs the internet and a few minutes."
            )
            built = self._call(
                ["build", "--pull", "-t", self.image, "-f", str(dockerfile), str(self.repo_root)],
                timeout=self.build_timeout,
                output_limit=64 * 1024,
            )
            self.actions.append(f"built {self.image}")
            if built is None or built.timed_out or built.returncode != 0 or not self._image_present():
                detail = ""
                if built is not None:
                    detail = (built.stderr or built.stdout or "").strip().splitlines()[-1:] or [""]
                    detail = detail[0][:200]
                return self._failed(
                    "I tried to build the sandbox image but it failed"
                    + (f" ({detail})" if detail else "")
                    + ". Check your internet connection and Docker, then ask me again."
                )

        return SandboxSetupResult(
            status=SandboxSetupStatus.COMPLETED,
            summary=(
                "The code sandbox is ready"
                + (f" ({'; '.join(self.actions)})" if self.actions else "")
                + ". Continuing your task."
            ),
            metadata=self._metadata(),
        )

    # ------------------------------------------------------------------
    @property
    def _product(self) -> str:
        return "Docker Desktop" if self.runtime_name == "docker" else "Podman"

    def _call(self, arguments, *, timeout, output_limit=4096):
        try:
            return self._run([self.executable, *arguments], timeout=timeout, output_limit=output_limit)
        except (SandboxRuntimeError, OSError):
            return None

    def _engine_ready(self) -> bool:
        result = self._call(["info", "--format", "{{.ServerVersion}}"], timeout=15.0)
        return bool(
            result is not None
            and not result.timed_out
            and result.returncode == 0
            and str(result.stdout or "").strip()
        )

    def _image_present(self) -> bool:
        result = self._call(["image", "inspect", self.image, "--format", "{{.Id}}"], timeout=15.0)
        return bool(result is not None and not result.timed_out and result.returncode == 0)

    def _start_engine(self) -> bool:
        if self.runtime_name == "podman":
            result = self._call(["machine", "start"], timeout=self.startup_timeout)
            self.actions.append("started the Podman machine")
            return result is not None
        # 1. Docker's own command (Docker Desktop 4.37+): no path needed.
        result = self._call(["desktop", "start", "--detach"], timeout=30.0)
        if result is not None and not result.timed_out and result.returncode == 0:
            self.actions.append("started Docker Desktop (docker desktop start)")
            return True
        self.observations.append("`docker desktop start` is unavailable")
        if self.platform.startswith("win"):
            # 2. The executable, found next to the CLI / in the registry / defaults.
            for path in _docker_desktop_candidates(getattr(self, "executable", None)):
                if path.is_file():
                    self._launch([str(path)])
                    self.actions.append(f"started Docker Desktop ({path})")
                    return True
            # 3. The same Start-menu lookup ASTA uses for "open <app>".
            target = self._installed_app_target("Docker Desktop")
            if target:
                self._open_target(target)
                self.actions.append("started Docker Desktop (Start menu)")
                return True
            return False
        if self.platform == "darwin":
            self._launch(["open", "-a", "Docker"])
            self.actions.append("started Docker Desktop")
            return True
        self._launch(["systemctl", "--user", "start", "docker-desktop"])
        self.actions.append("started Docker Desktop")
        return True

    def _wait_for_engine(self) -> bool:
        deadline = self.clock() + self.startup_timeout
        while self.clock() < deadline:
            self.sleep(3.0)
            if self._engine_ready():
                return True
        return False

    def _installed_app_target(self, name: str) -> str | None:
        manager = self.application_manager
        resolve = getattr(manager, "resolve", None)
        if not callable(resolve):
            return None
        try:
            record = resolve(name)
        except Exception:
            return None
        target = str(getattr(record, "launch_target", "") or "").strip()
        if not target or "docker" not in str(getattr(record, "name", "")).lower():
            return None
        return target

    @staticmethod
    def _shell_open(target: str) -> None:
        if os.name == "nt":
            try:
                os.startfile(target)  # type: ignore[attr-defined]
                return
            except OSError:
                pass
        SandboxSetupOperator._detached_launch(["explorer.exe", target] if os.name == "nt" else [target])

    @staticmethod
    def _detached_launch(command: list[str]) -> None:
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                shell=False,
                creationflags=flags,
            )
        except OSError:
            pass

    def _metadata(self) -> dict[str, Any]:
        return {"observed": list(self.observations), "actions": list(self.actions)}

    def _failed(self, summary: str, *, fixable: bool = True) -> SandboxSetupResult:
        metadata = self._metadata()
        metadata["fixable"] = fixable
        return SandboxSetupResult(status=SandboxSetupStatus.FAILED, summary=summary, metadata=metadata)
