import os
import platform
import subprocess
from pathlib import Path
from typing import Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


class OpenScreenshotTool(Tool):
    """Open the most recently captured A.S.T.A. screenshot."""

    def __init__(
        self,
        screenshot_dir: str | os.PathLike = "runtime/screenshots",
        opener: Callable[[Path], None] | None = None,
    ):
        self.screenshot_dir = Path(screenshot_dir)
        self._opener = opener

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.open_screenshot",
            description="Open the most recently captured screenshot.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["open_screenshot"],
                "action_aliases": ["show_screenshot", "open_latest_screenshot"],
                "category": "vision",
            },
        )

    def _latest_screenshot(self) -> Path | None:
        if not self.screenshot_dir.exists():
            return None

        files = [
            path
            for path in self.screenshot_dir.glob("asta_*.png")
            if path.is_file()
        ]
        if not files:
            return None

        return max(
            files,
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )

    def _open_path(self, path: Path) -> None:
        if self._opener is not None:
            self._opener(path)
            return

        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))
            return
        if system == "Darwin":
            subprocess.Popen(["open", str(path)])
            return
        if system == "Linux":
            subprocess.Popen(["xdg-open", str(path)])
            return

        raise RuntimeError(f"Unsupported platform: {system}")

    def execute(self, request: ToolRequest) -> ToolResult:
        import time

        start = time.perf_counter()
        latest = self._latest_screenshot()
        if latest is None:
            return ToolResult(
                success=False,
                tool=request.tool,
                error="No captured screenshots were found.",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        try:
            self._open_path(latest)
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                output={"path": str(latest.resolve())},
                error=f"Failed to open screenshot: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        return ToolResult(
            success=True,
            tool=request.tool,
            output={"path": str(latest.resolve()), "opened": True},
            duration_seconds=time.perf_counter() - start,
            metadata={"request_id": request.request_id},
        )
