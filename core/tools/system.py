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
        resolve_reference = getattr(self.application_manager, "resolve_reference", None)
        if callable(resolve_reference):
            try:
                # "open it" -> the app just opened or just mentioned.
                target = str(resolve_reference(target) or target).strip()
            except ApplicationResolutionError:
                pass
        resolved_target = None
        try:
            profile_launch = self._browser_profile_command(target)
            if profile_launch:
                # Chromium browsers opened without a profile show "Who's
                # using Chrome?"; open the last used profile directly.
                resolved_target = profile_launch[0]
                subprocess.Popen(
                    profile_launch,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                resolved_target = self.resolve_target(target)
                self._open(resolved_target)
        except ApplicationResolutionError as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                output=self._failure_output(target, None),
                error=str(exc),
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                output=self._failure_output(target, resolved_target),
                error=f"Application '{target}' not found on this system.",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                output=self._failure_output(target, resolved_target),
                error=f"Failed to open '{target}': {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
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
            metadata={"request_id": request.request_id},
        )

    @staticmethod
    def _browser_profile_command(target: str):
        if os.getenv("ASTA_BROWSER_PROFILE_LAUNCH", "1").strip().lower() in {"0", "false", "no", "off"}:
            return None
        from core.tools.browser import _find_executable, browser_key, chromium_profile

        key = browser_key(target)
        if key is None or key == "firefox":
            return None
        executable = _find_executable(key)
        profile = chromium_profile(key) if executable else None
        if not executable or not profile:
            return None
        return [executable, f"--profile-directory={profile}"]

    @staticmethod
    def _failure_output(target, resolved_target):
        output = {"target": target}
        if resolved_target is not None:
            output["resolved_target"] = resolved_target
        return output

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
            try:
                os.startfile(target)  # type: ignore[attr-defined]
            except OSError:
                # Microsoft Store/UWP applications resolve to shell:AppsFolder
                # targets. If the shell handoff through startfile fails, let
                # Explorer perform the same shell activation as a second native
                # path before surfacing the failure to task recovery.
                if str(target).lower().startswith("shell:appsfolder\\"):
                    explorer = shutil.which("explorer.exe") or shutil.which("explorer")
                    if explorer:
                        subprocess.Popen(
                            [explorer, target],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=getattr(
                                subprocess,
                                "CREATE_NO_WINDOW",
                                0,
                            ),
                        )
                        return
                raise
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
