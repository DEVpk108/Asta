"""Open a project's program on the user's desktop (games, GUIs, interactive apps).

The command sandbox has no screen, keyboard or extra packages, so a Snake game
"run" there either hangs until the timeout or crashes. With the user's go-ahead
(option A, chosen by the user), this tool starts the program on the real
desktop from the active workspace so the user can see and play it. Because it
runs outside the sandbox it is ALWAYS confirmed first (risk "critical"), and
only .py files inside the active workspace can be opened.

It watches the first few seconds: an immediate crash (for example
``ModuleNotFoundError: No module named 'pygame'``) is reported as a failure
with the error text, which the edit-fix loop uses to repair the code. A program
that is still running, or that finished cleanly, counts as opened.

Interpreter: the workspace's own virtual environment (.venv / venv) when it
has one, else ASTA_PROJECT_PYTHON, else ``python`` on PATH, else ASTA's own.
ASTA_PROGRAM_SETTLE_SECONDS (default 3) sets how long it watches for a crash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from core.contracts import ToolDefinition, ToolRequest, ToolResult

from .base import Tool
from .workspace import (
    WorkspacePathError,
    _is_sensitive_path,
    resolve_workspace_path,
    resolve_workspace_root,
)

MAX_PROGRAM_ARGUMENTS = 32
_STDERR_TAIL = 4000


def _settle_seconds() -> float:
    try:
        return min(15.0, max(0.5, float(os.getenv("ASTA_PROGRAM_SETTLE_SECONDS", "3"))))
    except (TypeError, ValueError):
        return 3.0


def project_python(root: Path) -> str:
    for candidate in (
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    configured = os.getenv("ASTA_PROJECT_PYTHON", "").strip()
    if configured:
        return configured
    return shutil.which("python") or shutil.which("python3") or sys.executable


class OpenProjectProgramTool(Tool):
    def __init__(self, workspace_manager=None, *, popen=subprocess.Popen, settle_seconds: float | None = None):
        self.workspace_manager = workspace_manager
        self._popen = popen
        self._settle = settle_seconds

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="project.open_program",
            description=(
                "Open a Python program from the active workspace on the user's desktop so they "
                "can see and use it (games, windows, interactive apps). Runs outside the sandbox "
                "and always asks the user first."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "description": "Workspace-relative .py file."},
                    "arguments": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_PROGRAM_ARGUMENTS},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            risk_level="critical",
            metadata={"actions": ["open_program"], "category": "project", "workspace_scoped": True},
        )

    def _result(self, request, start, *, success, output=None, error=None) -> ToolResult:
        return ToolResult(
            success=success, tool=request.tool, output=output, error=error,
            duration_seconds=time.perf_counter() - start,
            metadata={"request_id": request.request_id},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        arguments = request.arguments.get("arguments") or []
        if not isinstance(arguments, list) or len(arguments) > MAX_PROGRAM_ARGUMENTS or not all(
            isinstance(item, str) and "\x00" not in item for item in arguments
        ):
            return self._result(request, start, success=False, error="Program arguments must be a short list of strings.")
        try:
            root = resolve_workspace_root(self.workspace_manager)
            program = resolve_workspace_path(root, request.arguments.get("path"))
            if program.suffix.lower() not in {".py", ".pyw"}:
                raise WorkspacePathError("Only Python programs (.py) from the workspace can be opened.")
            if _is_sensitive_path(root, program) or not program.is_file():
                raise WorkspacePathError(f"{request.arguments.get('path')} isn't a program file in the workspace.")
        except (OSError, RuntimeError, ValueError, WorkspacePathError) as exc:
            return self._result(request, start, success=False, error=str(exc))

        relative = program.relative_to(root).as_posix()
        python = project_python(root)
        log = tempfile.NamedTemporaryFile(prefix="asta-program-", suffix=".log", delete=False)
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
        try:
            process = self._popen(
                [python, relative, *arguments],
                cwd=str(root),
                stdin=subprocess.DEVNULL if flags == 0 else None,
                stderr=log,
                creationflags=flags,
                **({} if os.name == "nt" else {"start_new_session": True}),
            )
        except Exception as exc:
            log.close()
            return self._result(request, start, success=False,
                                error=f"Couldn't start {relative}: {type(exc).__name__}: {exc}")

        settle = self._settle if self._settle is not None else _settle_seconds()
        deadline = time.monotonic() + settle
        returncode = process.poll()
        while returncode is None and time.monotonic() < deadline:
            time.sleep(0.1)
            returncode = process.poll()
        log.close()
        try:
            stderr = Path(log.name).read_text(encoding="utf-8", errors="replace")[-_STDERR_TAIL:]
        except OSError:
            stderr = ""
        output = {
            "path": relative,
            "command": f"python {relative}" + (" " + " ".join(arguments) if arguments else ""),
            "pid": getattr(process, "pid", None),
            "state": "running" if returncode is None else "finished",
            "returncode": returncode,
            "stderr": stderr,
            "launched": True,
        }
        if returncode not in (None, 0):
            last = next((line for line in reversed(stderr.strip().splitlines()) if line.strip()), "")
            return self._result(
                request, start, success=False, output=output,
                error=f"{relative} crashed right after starting (exit {returncode})" + (f": {last}" if last else "."),
            )
        return self._result(request, start, success=True, output=output)
