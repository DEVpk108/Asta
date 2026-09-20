import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

from core.applications import ApplicationManager, ApplicationResolutionError
from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


class OpenApplicationTool(Tool):
    """Open a path, URL, or discovered host application."""

    def __init__(self, application_manager=None):
        self.application_manager = application_manager or ApplicationManager()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.open_application",
            description="Open a local application, file, URL, or discovered application.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string", "minLength": 1}},
                "required": ["target"],
            },
            risk_level="medium",
            requires_confirmation=False,
            timeout_seconds=10.0,
            metadata={
                "actions": ["open"],
                "category": "system",
                "platforms": ["windows", "macos", "linux"],
                "application_discovery": True,
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")

        if not isinstance(target, str) or not target.strip():
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error="Argument 'target' must be a non-empty string.",
            )

        target = target.strip()
        try:
            resolved_target = self.resolve_target(target)
            self._open(resolved_target)
        except ApplicationResolutionError as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=str(exc),
                duration_seconds=time.perf_counter() - start,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=f"Application '{target}' not found on this system.",
                duration_seconds=time.perf_counter() - start,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=f"Failed to open '{target}': {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
            )

        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={
                "target": target,
                "resolved_target": resolved_target,
                "opened": True,
            },
            duration_seconds=time.perf_counter() - start,
        )

    def resolve_target(self, target: str) -> str:
        """Resolve an application through generic host discovery."""
        if os.name == "nt":
            if os.path.exists(target) or self._looks_like_uri(target):
                return target

            executable = shutil.which(target)
            if executable:
                return executable

            return self.application_manager.resolve(target).launch_target

        executable = shutil.which(target)
        if executable:
            return executable

        if os.path.exists(target) or self._looks_like_uri(target):
            return target

        raise FileNotFoundError(f"Application or URI target not found: {target}")

    @staticmethod
    def _looks_like_uri(target: str) -> bool:
        parsed = urlparse(target)
        return bool(parsed.scheme and (parsed.netloc or target.endswith(":")))

    @staticmethod
    def _open(target: str) -> None:
        if os.name == "nt":
            os.startfile(target)  # type: ignore[attr-defined]
            return

        if _platform_is_macos():
            subprocess.Popen(["open", target])
            return

        if _platform_is_linux():
            subprocess.Popen(["xdg-open", target])
            return

        executable = shutil.which(target)
        if executable:
            subprocess.Popen([executable])
            return

        raise OSError(
            "No supported native application opener is available for this platform."
        )


def _platform_is_macos() -> bool:
    return os.uname().sysname == "Darwin" if hasattr(os, "uname") else False


def _platform_is_linux() -> bool:
    return os.uname().sysname == "Linux" if hasattr(os, "uname") else False
