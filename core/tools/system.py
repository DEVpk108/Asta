import os
import shutil
import subprocess
import time

from core.contracts import ToolDefinition, ToolRequest, ToolResult

from core.tools.base import Tool


class OpenApplicationTool(Tool):
    """Open a local application or explicitly supplied file/URL.

    The tool intentionally avoids shell execution. It uses the platform's
    native file/application opener where available and falls back to an
    executable resolved from PATH.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.open_application",
            description="Open a local application, file, or URL.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": ["target"],
            },
            risk_level="medium",
            requires_confirmation=False,
            timeout_seconds=10.0,
            metadata={
                "actions": ["open"],
                "category": "system",
                "platforms": ["windows", "macos", "linux"],
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
            self._open(target)
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
            output={"target": target, "opened": True},
            duration_seconds=time.perf_counter() - start,
        )

    @staticmethod
    def _open(target: str) -> None:
        if os.name == "nt":
            # os.startfile delegates to Windows shell associations without
            # constructing a shell command ourselves.
            os.startfile(target)  # type: ignore[attr-defined]
            return

        if sys_platform_is_macos():
            subprocess.Popen(["open", target])
            return

        if sys_platform_is_linux():
            subprocess.Popen(["xdg-open", target])
            return

        executable = shutil.which(target)
        if executable:
            subprocess.Popen([executable])
            return

        raise OSError(
            "No supported native application opener is available for this platform."
        )


def sys_platform_is_macos() -> bool:
    return os.uname().sysname == "Darwin" if hasattr(os, "uname") else False


def sys_platform_is_linux() -> bool:
    return os.uname().sysname == "Linux" if hasattr(os, "uname") else False
