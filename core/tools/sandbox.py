"""Fail-closed Python command execution in an isolated container snapshot."""

from __future__ import annotations

import math
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool
from core.tools.workspace import (
    WorkspacePathError,
    _is_sensitive_parts,
    resolve_workspace_path,
    resolve_workspace_root,
)


DEFAULT_SANDBOX_IMAGE = "asta-python-sandbox:1"
MAX_COMMAND_ARGUMENTS = 256
MAX_COMMAND_ARGUMENT_BYTES = 32 * 1024
MAX_OUTPUT_BYTES_PER_STREAM = 128 * 1024
MAX_SNAPSHOT_FILES = 10_000
MAX_SNAPSHOT_DIRECTORIES = 20_000
MAX_SNAPSHOT_FILE_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
MAX_COMMAND_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class SandboxProcessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class SandboxRuntimeError(RuntimeError):
    """The local container runtime is missing or could not be started."""


def _resolve_runtime_executable(
    runtime_name: str,
    *,
    is_windows: bool | None = None,
) -> str | None:
    """Resolve the native CLI binary, never a Windows .cmd/.bat shim."""
    windows = os.name == "nt" if is_windows is None else bool(is_windows)
    candidates = (
        (f"{runtime_name}.exe", runtime_name)
        if windows
        else (runtime_name,)
    )
    for candidate in candidates:
        executable = shutil.which(candidate)
        if not executable:
            continue
        if windows and not executable.lower().endswith(".exe"):
            continue
        return executable
    return None


def _bounded_process(
    command: list[str],
    *,
    timeout: float,
    output_limit: int,
) -> SandboxProcessResult:
    """Run one container-runtime CLI call without buffering unbounded output."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except OSError as exc:
        raise SandboxRuntimeError(
            f"Could not start the container runtime: {type(exc).__name__}: {exc}"
        ) from exc

    buffers = [bytearray(), bytearray()]
    truncated = [False, False]

    def drain(index: int, stream) -> None:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            remaining = output_limit - len(buffers[index])
            if remaining > 0:
                buffers[index].extend(chunk[:remaining])
            if len(chunk) > max(0, remaining):
                truncated[index] = True

    readers = [
        threading.Thread(target=drain, args=(0, process.stdout), daemon=True),
        threading.Thread(target=drain, args=(1, process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        returncode = process.wait()
    finally:
        for reader in readers:
            reader.join(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    return SandboxProcessResult(
        returncode=int(returncode),
        stdout=bytes(buffers[0]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers[1]).decode("utf-8", errors="replace"),
        timed_out=timed_out,
        stdout_truncated=truncated[0],
        stderr_truncated=truncated[1],
    )


class ContainerRuntime:
    """Small Docker/Podman CLI adapter with bounded output capture."""

    def __init__(self, runtime_name: str | None = None):
        selected = (runtime_name or os.getenv("ASTA_CONTAINER_RUNTIME", "docker")).strip().lower()
        if selected not in {"docker", "podman"}:
            raise ValueError("ASTA_CONTAINER_RUNTIME must be either 'docker' or 'podman'.")
        self.runtime_name = selected

    def run(
        self,
        arguments: list[str],
        *,
        timeout: float,
        output_limit: int = MAX_OUTPUT_BYTES_PER_STREAM,
    ) -> SandboxProcessResult:
        executable = _resolve_runtime_executable(self.runtime_name)
        if not executable:
            raise SandboxRuntimeError(
                f"A native {self.runtime_name} CLI executable was not found on PATH. "
                "Install/start Docker or Podman and ensure its .exe is available."
            )
        return _bounded_process(
            [executable, *arguments],
            timeout=timeout,
            output_limit=output_limit,
        )


def _safe_target(value: object) -> tuple[str, list[str]] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    target = value.strip()
    # Older planners sometimes provide a host interpreter path. Only its
    # basename is considered; the path is never executed on the host.
    basename = target.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename in {"python", "python3", "python.exe", "python3.exe"}:
        return "python", ["/usr/local/bin/python"]
    if basename in {"pytest", "pytest.exe"}:
        return "pytest", ["/usr/local/bin/python", "-m", "pytest"]
    return None


def _safe_arguments(value: object) -> list[str] | None:
    if not isinstance(value, list) or len(value) > MAX_COMMAND_ARGUMENTS:
        return None
    if not all(isinstance(item, str) and "\x00" not in item for item in value):
        return None
    if any(len(item) > 8192 for item in value):
        return None
    if sum(len(item.encode("utf-8", errors="replace")) for item in value) > MAX_COMMAND_ARGUMENT_BYTES:
        return None
    return value


def _copy_workspace_snapshot(root: Path, destination: Path) -> tuple[int, int]:
    """Copy only ordinary, non-sensitive project files into a bounded snapshot."""
    copied_files = 0
    copied_bytes = 0
    visited_directories = 0
    traversal_errors: list[OSError] = []

    def on_walk_error(error: OSError) -> None:
        traversal_errors.append(error)

    for current, directories, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=on_walk_error,
    ):
        source_dir = Path(current)
        visited_directories += 1
        if visited_directories > MAX_SNAPSHOT_DIRECTORIES:
            raise WorkspacePathError("Sandbox snapshot exceeds its directory-count limit.")
        relative_dir = source_dir.relative_to(root)
        destination_dir = destination / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            destination_dir.chmod(0o755)
        except OSError:
            pass

        allowed_directories = []
        for name in sorted(directories):
            source_child = source_dir / name
            relative_parts = (*relative_dir.parts, name)
            try:
                if (
                    source_child.is_symlink()
                    or source_child.is_mount()
                    or _is_sensitive_parts(relative_parts)
                ):
                    continue
            except OSError:
                continue
            allowed_directories.append(name)
        directories[:] = allowed_directories

        for filename in sorted(filenames):
            source_file = source_dir / filename
            relative_file = source_file.relative_to(root)
            if _is_sensitive_parts(relative_file.parts) or source_file.is_symlink():
                continue
            try:
                info = source_file.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_size > MAX_SNAPSHOT_FILE_BYTES:
                raise WorkspacePathError(
                    f"Sandbox snapshot refused an oversized file: {relative_file.as_posix()}"
                )
            copied_files += 1
            if (
                copied_files > MAX_SNAPSHOT_FILES
                or copied_bytes + info.st_size > MAX_SNAPSHOT_BYTES
            ):
                raise WorkspacePathError(
                    "Sandbox snapshot exceeds its file-count or size limit."
                )
            destination_file = destination / relative_file
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                no_follow = getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(source_file, os.O_RDONLY | no_follow)
                with os.fdopen(descriptor, "rb") as source_stream:
                    opened_info = os.fstat(source_stream.fileno())
                    if (
                        not stat.S_ISREG(opened_info.st_mode)
                        or opened_info.st_dev != info.st_dev
                        or opened_info.st_ino != info.st_ino
                    ):
                        continue
                    file_bytes = 0
                    with destination_file.open("xb") as destination_stream:
                        while True:
                            chunk = source_stream.read(64 * 1024)
                            if not chunk:
                                break
                            file_bytes += len(chunk)
                            if (
                                file_bytes > MAX_SNAPSHOT_FILE_BYTES
                                or copied_bytes + file_bytes > MAX_SNAPSHOT_BYTES
                            ):
                                raise WorkspacePathError(
                                    "Sandbox snapshot exceeded its file-size or total-size limit."
                                )
                            destination_stream.write(chunk)
                    copied_bytes += file_bytes
                destination_file.chmod(0o644)
            except WorkspacePathError:
                raise
            except OSError as exc:
                raise WorkspacePathError(
                    f"Could not copy workspace file into the sandbox snapshot: "
                    f"{relative_file.as_posix()}"
                ) from exc

    if traversal_errors:
        raise WorkspacePathError(
            "The active workspace could not be completely read for a safe sandbox snapshot."
        ) from traversal_errors[0]
    return copied_files, copied_bytes


def _docker_mount_source(path: Path) -> str:
    source = path.resolve(strict=True).as_posix()
    # Docker's --mount syntax uses commas as option separators.
    if "," in source:
        raise WorkspacePathError(
            "The temporary sandbox path contains a comma and cannot be mounted safely."
        )
    return source


def _tool_result(
    request: ToolRequest,
    *,
    success: bool,
    output=None,
    error: str | None = None,
    start: float,
) -> ToolResult:
    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


class RunCommandTool(Tool):
    """Run Python/pytest against a sanitized, read-only project snapshot."""

    def __init__(self, workspace_manager=None, *, runtime=None, image: str | None = None):
        self.workspace_manager = workspace_manager
        self.runtime = runtime or ContainerRuntime()
        self.image = (image or os.getenv("ASTA_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)).strip()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.run_command",
            description=(
                "Run an explicitly supplied Python or pytest command in a local "
                "network-disabled container against a sanitized, read-only snapshot "
                "of the active project workspace. Workspace changes are discarded; "
                "the host executable, commonly named secrets, virtual environments, "
                "and VCS metadata are not mounted."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Python or pytest only; host interpreter paths are reduced to their basename.",
                    },
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": MAX_COMMAND_ARGUMENTS,
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional directory relative to the active workspace.",
                    },
                },
                "required": ["target"],
                "additionalProperties": False,
            },
            risk_level="critical",
            timeout_seconds=MAX_COMMAND_TIMEOUT_SECONDS,
            metadata={"actions": ["run"], "category": "system", "workspace_scoped": True},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _safe_target(request.arguments.get("target"))
        arguments = _safe_arguments(request.arguments.get("arguments", []))
        if target is None:
            return _tool_result(
                request,
                success=False,
                error="Only Python and pytest commands are available in the command sandbox.",
                start=start,
            )
        if arguments is None:
            return _tool_result(
                request,
                success=False,
                error="Command arguments must be a bounded list of strings without NUL bytes.",
                start=start,
            )
        if not self.image or len(self.image) > 200 or self.image.startswith("-") or any(
            char.isspace() or char == "," for char in self.image
        ):
            return _tool_result(
                request,
                success=False,
                error="ASTA_SANDBOX_IMAGE is invalid.",
                start=start,
            )

        try:
            root = resolve_workspace_root(self.workspace_manager)
            requested_cwd = request.arguments.get("cwd", ".")
            if requested_cwd is None or requested_cwd == "":
                requested_cwd = "."
            working_directory = resolve_workspace_path(
                root,
                requested_cwd,
                allow_root=True,
            )
            if not working_directory.is_dir():
                raise WorkspacePathError("The requested workspace working directory is not a directory.")
            working_relative = working_directory.relative_to(root)
            if _is_sensitive_parts(working_relative.parts):
                raise WorkspacePathError(
                    "The requested workspace directory is not available in a sandbox snapshot."
                )
            relative_cwd = working_relative.as_posix()
            if relative_cwd == ".":
                relative_cwd = ""
        except (OSError, RuntimeError, WorkspacePathError, ValueError) as exc:
            return _tool_result(request, success=False, error=str(exc), start=start)

        raw_timeout = request.timeout_seconds
        if isinstance(raw_timeout, bool):
            return _tool_result(
                request,
                success=False,
                error="Command timeout must be a finite positive number.",
                start=start,
            )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            timeout = 0.0
        if not math.isfinite(timeout) or timeout <= 0:
            return _tool_result(
                request,
                success=False,
                error="Command timeout must be a finite positive number.",
                start=start,
            )
        timeout = min(timeout, MAX_COMMAND_TIMEOUT_SECONDS)

        container_name = f"asta-sandbox-{uuid4().hex}"
        snapshot_path = None
        created = False
        cleanup_needed = False
        timed_out = False
        cleanup_error = None
        result = None
        try:
            snapshot_path = Path(tempfile.mkdtemp(prefix="asta-sandbox-"))
            _copy_workspace_snapshot(root, snapshot_path)
            snapshot_cwd = snapshot_path / relative_cwd
            if not snapshot_cwd.is_dir():
                raise WorkspacePathError(
                    "The requested working directory was omitted from the safe sandbox snapshot."
                )
            container_cwd = "/workspace" + (f"/{relative_cwd}" if relative_cwd else "")
            source = _docker_mount_source(snapshot_path)
            canonical_target, executable = target
            container_entrypoint = executable[0]
            container_command = [*executable[1:], *arguments]

            create_arguments = [
                "create",
                "--pull=never",
                "--name",
                container_name,
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=64",
                "--memory=1g",
                "--memory-swap=1g",
                "--cpus=1",
                "--ulimit",
                "core=0",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001",
                "--mount",
                f"type=bind,source={source},target=/workspace,readonly",
                "--workdir",
                container_cwd,
                "--user=10001:10001",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "PYTEST_ADDOPTS=-p no:cacheprovider",
                "--entrypoint",
                container_entrypoint,
                self.image,
                *container_command,
            ]
            create_result = self.runtime.run(
                create_arguments,
                timeout=10.0,
                output_limit=4096,
            )
            created = create_result.returncode == 0
            cleanup_needed = created or create_result.timed_out
            if create_result.timed_out or not created:
                detail = (create_result.stderr or create_result.stdout).strip()[:700]
                error = (
                    "Sandbox container creation timed out."
                    if create_result.timed_out
                    else (
                        "Sandbox container creation failed. Confirm Docker or Podman is running "
                        "and the image is already built locally; ASTA never pulls images at runtime."
                    )
                )
                if detail:
                    error += f" Runtime: {detail}"
                result = _tool_result(request, success=False, error=error, start=start)
            else:
                start_result = self.runtime.run(
                    ["start", "--attach", container_name],
                    timeout=timeout,
                    output_limit=MAX_OUTPUT_BYTES_PER_STREAM,
                )
                timed_out = start_result.timed_out
                output = {
                    "target": canonical_target,
                    "returncode": start_result.returncode,
                    "stdout": start_result.stdout,
                    "stderr": start_result.stderr,
                    "sandbox": {
                        "runtime": getattr(self.runtime, "runtime_name", "injected"),
                        "network": "none",
                        "workspace_snapshot": "read-only",
                        "output_truncated": (
                            start_result.stdout_truncated or start_result.stderr_truncated
                        ),
                    },
                }
                if start_result.stdout_truncated:
                    output["stdout"] += "\n[stdout truncated at the configured safety limit]"
                if start_result.stderr_truncated:
                    output["stderr"] += "\n[stderr truncated at the configured safety limit]"
                result = _tool_result(
                    request,
                    output=output,
                    success=(
                        not timed_out
                        and start_result.returncode == 0
                    ),
                    error=(
                        f"Command timed out after {timeout:.1f}s; sandbox cleanup was requested."
                        if timed_out
                        else (
                            None
                            if start_result.returncode == 0
                            else f"Sandbox command exited with code {start_result.returncode}."
                        )
                    ),
                    start=start,
                )
        except (WorkspacePathError, OSError, SandboxRuntimeError, ValueError) as exc:
            result = _tool_result(
                request,
                success=False,
                error=f"Sandbox unavailable: {type(exc).__name__}: {exc}",
                start=start,
            )
        finally:
            if cleanup_needed:
                if timed_out:
                    try:
                        self.runtime.run(
                            ["kill", container_name],
                            timeout=10.0,
                            output_limit=4096,
                        )
                    except Exception:
                        pass
                try:
                    cleanup = self.runtime.run(
                        ["rm", "--force", container_name],
                        timeout=10.0,
                        output_limit=4096,
                    )
                    if cleanup.timed_out or cleanup.returncode != 0:
                        cleanup_error = "The sandbox container could not be confirmed removed."
                except Exception:
                    cleanup_error = "The sandbox container could not be confirmed removed."
            if snapshot_path is not None:
                shutil.rmtree(snapshot_path, ignore_errors=True)
        if result is None:
            result = _tool_result(
                request,
                success=False,
                error="Sandbox execution ended without a result.",
                start=start,
            )
        if cleanup_error:
            result = ToolResult(
                success=False,
                tool=result.tool,
                output=result.output,
                error=cleanup_error,
                duration_seconds=result.duration_seconds,
                metadata=result.metadata,
            )
            print(f"[Sandbox] {cleanup_error} ({container_name})", flush=True)
        return result