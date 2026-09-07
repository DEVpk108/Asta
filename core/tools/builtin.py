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


class LaunchApplicationTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.launch_application",
            description="Launch a local application by executable name or path.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string", "minLength": 1}},
                "required": ["target"],
            },
            risk_level="medium",
            metadata={"actions": ["launch"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")
        if not isinstance(target, str) or not target.strip():
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)
        target = target.strip()
        executable = shutil.which(target) or (target if os.path.exists(target) else None)
        if not executable:
            return _result(request, False, error=f"Application not found: {target}", start=start)
        try:
            subprocess.Popen([executable], start_new_session=True)
        except Exception as exc:
            return _result(request, False, error=f"Failed to launch '{target}': {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output={"target": target, "launched": True}, start=start)


class StartProcessTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.start_process",
            description="Start a local executable process with an optional argument list.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "minLength": 1},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target"],
            },
            risk_level="high",
            metadata={"actions": ["start"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")
        args = request.arguments.get("arguments", [])
        if not isinstance(target, str) or not target.strip():
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
            name="system.run_command",
            description="Run an explicitly supplied local command without invoking a shell.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "minLength": 1},
                    "arguments": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target"],
            },
            risk_level="critical",
            metadata={"actions": ["run"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")
        args = request.arguments.get("arguments", [])
        if not isinstance(target, str) or not target.strip():
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
        return _result(
            request,
            completed.returncode == 0,
            output={"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
            error=None if completed.returncode == 0 else f"Command exited with code {completed.returncode}.",
            start=start,
        )


class StopProcessTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.stop_process",
            description="Terminate a local process by PID.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string", "minLength": 1}},
                "required": ["target"],
            },
            risk_level="high",
            metadata={"actions": ["stop"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")
        try:
            pid = int(target)
            if pid <= 0:
                raise ValueError
        except (TypeError, ValueError):
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


class CloseApplicationTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.close_application",
            description="Close a local application using a supplied process ID.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string", "minLength": 1}},
                "required": ["target"],
            },
            risk_level="high",
            metadata={"actions": ["close"], "category": "system"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        return StopProcessTool().execute(request)


class ScreenshotTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.screenshot",
            description="Capture a screenshot using an installed screenshot backend.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            risk_level="low",
            metadata={"actions": ["screenshot"], "category": "vision"},
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
            name=f"audio.{self.action}",
            description=f"{self.action.capitalize()} system audio.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            risk_level="low",
            metadata={"actions": [self.action], "category": "audio", "platforms": [platform.system().lower()]},
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
