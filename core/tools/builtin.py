import os
import platform
import shutil
import signal
import subprocess
import time
from typing import Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


def _result(request: ToolRequest, success: bool, *, output=None, error=None, start: float) -> ToolResult:
    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


def _validated_target(request: ToolRequest):
    target = request.arguments.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


class LaunchApplicationTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.launch_application", description="Launch a local application by executable name or path.",
            input_schema={"type": "object", "properties": {"target": {"type": "string", "minLength": 1}}, "required": ["target"]},
            risk_level="medium", metadata={"actions": ["launch"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _validated_target(request)
        if target is None:
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)
        executable = shutil.which(target) or (target if os.path.exists(target) else None)
        if not executable:
            return _result(request, False, error=f"Application not found: {target}", start=start)
        try:
            process = subprocess.Popen([executable], start_new_session=True)
        except Exception as exc:
            return _result(request, False, error=f"Failed to launch '{target}': {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output={"target": target, "pid": process.pid, "launched": True}, start=start)


class StartProcessTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.start_process", description="Start a local executable process with an optional argument list.",
            input_schema={"type": "object", "properties": {"target": {"type": "string", "minLength": 1}, "arguments": {"type": "array", "items": {"type": "string"}}}, "required": ["target"]},
            risk_level="high", metadata={"actions": ["start"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _validated_target(request)
        args = request.arguments.get("arguments", [])
        if target is None:
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            return _result(request, False, error="Argument 'arguments' must be a list of strings.", start=start)
        executable = shutil.which(target) or (target if os.path.exists(target) else None)
        if not executable:
            return _result(request, False, error=f"Executable not found: {target}", start=start)
        try:
            process = subprocess.Popen([executable, *args], start_new_session=True)
        except Exception as exc:
            return _result(request, False, error=f"Failed to start '{target}': {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output={"target": target, "pid": process.pid}, start=start)


class RunCommandTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.run_command", description="Run an explicitly supplied local executable without invoking a shell.",
            input_schema={"type": "object", "properties": {"target": {"type": "string", "minLength": 1}, "arguments": {"type": "array", "items": {"type": "string"}}}, "required": ["target"]},
            risk_level="critical", metadata={"actions": ["run"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _validated_target(request)
        args = request.arguments.get("arguments", [])
        if target is None:
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            return _result(request, False, error="Argument 'arguments' must be a list of strings.", start=start)
        executable = shutil.which(target)
        if not executable:
            return _result(request, False, error=f"Command not found on PATH: {target}", start=start)
        try:
            completed = subprocess.run([executable, *args], capture_output=True, text=True, timeout=request.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return _result(request, False, error=f"Command timed out after {request.timeout_seconds:.1f}s.", start=start)
        except Exception as exc:
            return _result(request, False, error=f"Failed to run '{target}': {type(exc).__name__}: {exc}", start=start)
        return _result(request, completed.returncode == 0,
                       output={"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
                       error=None if completed.returncode == 0 else f"Command exited with code {completed.returncode}.", start=start)


class StopProcessTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.stop_process", description="Terminate a local process by PID.",
            input_schema={"type": "object", "properties": {"target": {"type": "string", "pattern": "^[1-9][0-9]*$"}}, "required": ["target"]},
            risk_level="high", metadata={"actions": ["stop"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _validated_target(request)
        try:
            pid = int(target) if target is not None else 0
            if pid <= 0:
                raise ValueError
        except ValueError:
            return _result(request, False, error="Argument 'target' must be a positive process ID.", start=start)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return _result(request, False, error=f"Process not found: {pid}", start=start)
        except PermissionError:
            return _result(request, False, error=f"Permission denied stopping process {pid}.", start=start)
        except Exception as exc:
            return _result(request, False, error=f"Failed to stop process {pid}: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output={"pid": pid, "stopped": True}, start=start)


WINDOWS_PROCESS_ALIASES = {
    "camera": ("WindowsCamera.exe", "WindowsCameraApp.exe"),
    "windows camera": ("WindowsCamera.exe", "WindowsCameraApp.exe"),
}


class CloseApplicationTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.close_application", description="Close a local application by process name or friendly application name.",
            input_schema={"type": "object", "properties": {"target": {"type": "string", "minLength": 1}}, "required": ["target"]},
            risk_level="high", metadata={"actions": ["close"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = _validated_target(request)
        if target is None:
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)

        try:
            system = platform.system()
            if system == "Windows":
                candidates = []
                normalized = target.lower()
                candidates.extend(WINDOWS_PROCESS_ALIASES.get(normalized, ()))
                candidates.append(target)
                if not target.lower().endswith(".exe"):
                    candidates.append(f"{target}.exe")

                seen = set()
                attempts = []
                for image_name in candidates:
                    if image_name.lower() in seen:
                        continue
                    seen.add(image_name.lower())
                    completed = subprocess.run(
                        ["taskkill", "/IM", image_name, "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_seconds,
                        check=False,
                    )
                    attempts.append(completed)
                    if completed.returncode == 0:
                        return _result(
                            request,
                            True,
                            output={"target": target, "process": image_name, "closed": True},
                            start=start,
                        )

                stderr = next((item.stderr.strip() for item in reversed(attempts) if item.stderr.strip()), "")
                stdout = next((item.stdout.strip() for item in reversed(attempts) if item.stdout.strip()), "")
                detail = stderr or stdout
                error = f"Application '{target}' was not closed."
                if detail:
                    error += f" {detail}"
                return _result(request, False, output={"target": target, "closed": False}, error=error, start=start)

            if system in {"Linux", "Darwin"}:
                completed = subprocess.run(["pkill", "-TERM", "-x", target], capture_output=True, text=True, timeout=request.timeout_seconds, check=False)
            else:
                return _result(request, False, error=f"Unsupported platform: {system}", start=start)
        except subprocess.TimeoutExpired:
            return _result(request, False, error=f"Close operation timed out after {request.timeout_seconds:.1f}s.", start=start)
        except Exception as exc:
            return _result(request, False, error=f"Failed to close '{target}': {type(exc).__name__}: {exc}", start=start)
        return _result(request, completed.returncode == 0, output={"target": target, "closed": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}, error=None if completed.returncode == 0 else f"Application '{target}' was not closed.", start=start)


class ScreenshotTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.screenshot", description="Capture a screenshot using the configured screenshot backend.",
            input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            risk_level="low", metadata={"actions": ["screenshot"], "category": "vision"},
        )

    def __init__(self, capture: Callable | None = None):
        self._capture = capture

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        if self._capture is None:
            return _result(request, False, error="No screenshot backend is configured yet.", start=start)
        try:
            output = self._capture()
        except Exception as exc:
            return _result(request, False, error=f"Screenshot failed: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output=output, start=start)


class AudioControlTool(Tool):
    def __init__(self, action: str, handler: Callable | None = None):
        if action not in {"mute", "unmute"}:
            raise ValueError("AudioControlTool action must be mute or unmute.")
        self.action = action
        self._handler = handler

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=f"audio.{self.action}", description=f"{self.action.capitalize()} system audio.",
            input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            risk_level="low", metadata={"actions": [self.action], "category": "audio", "platforms": [platform.system().lower()]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        if self._handler is None:
            return _result(request, False, error=f"No audio backend is configured for {self.action}.", start=start)
        try:
            output = self._handler()
        except Exception as exc:
            return _result(request, False, error=f"Audio control failed: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output=output, start=start)