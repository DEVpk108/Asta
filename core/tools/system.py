import os
import platform
import shutil
import subprocess
import time
from urllib.parse import urlparse

from core.contracts import ToolDefinition, ToolRequest, ToolResult

from core.tools.base import Tool


# Friendly application names that cannot reliably be launched by passing the
# spoken name directly to Windows. These are intentionally limited to known
# non-destructive application launch targets; arbitrary URI schemes are still
# accepted only when the user explicitly supplies one.
WINDOWS_APPLICATION_ALIASES = {
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "camera": "microsoft.windows.camera:",
    "windows camera": "microsoft.windows.camera:",
    "spotify": "spotify:",
}


class OpenApplicationTool(Tool):
    """Open a local application, file, URL, or known friendly app alias.

    The tool intentionally avoids shell execution. On Windows it first tries
    a known safe application alias, then an executable resolved from PATH,
    then the native Windows opener for paths/URIs. On POSIX platforms it uses
    the platform's native opener before falling back to PATH resolution.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="system.open_application",
            description=(
                "Open a local application, file, URL, or known application alias."
            ),
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
        resolved_target = self.resolve_target(target)

        try:
            self._open(resolved_target)
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=(
                    f"Failed to open '{target}': "
                    f"{type(exc).__name__}: {exc}"
                ),
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

    @classmethod
    def resolve_target(cls, target: str) -> str:
        """Resolve a friendly app name to a safe OS launch target."""
        normalized = target.strip().lower()

        if os.name == "nt":
            alias = WINDOWS_APPLICATION_ALIASES.get(normalized)
            if alias:
                return alias

            executable = shutil.which(target)
            if executable:
                return executable

            # Preserve Windows paths and explicitly supplied URIs for the
            # native opener. Do not reinterpret arbitrary unknown names.
            if os.path.exists(target) or cls._looks_like_uri(target):
                return target

            raise FileNotFoundError(
                f"Application or URI target not found: {target}"
            )

        executable = shutil.which(target)
        if executable:
            return executable

        if os.path.exists(target) or cls._looks_like_uri(target):
            return target

        raise FileNotFoundError(f"Application or URI target not found: {target}")

    @staticmethod
    def _looks_like_uri(target: str) -> bool:
        parsed = urlparse(target)
        return bool(parsed.scheme and (parsed.netloc or target.endswith(":")))

    @staticmethod
    def _open(target: str) -> None:
        if os.name == "nt":
            # os.startfile delegates to Windows shell associations / URI
            # handlers without constructing a shell command ourselves.
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
