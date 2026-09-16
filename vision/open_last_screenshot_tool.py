import os
import platform
import time
from pathlib import Path

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


class OpenLastScreenshotTool(Tool):
    """Open the most recently captured screenshot in the system viewer."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.open_last_screenshot",
            description="Open the most recently captured screenshot using the system image viewer.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["open_screenshot"],
                "action_aliases": ["open screenshot", "show screenshot"],
                "category": "vision",
            },
        )

    def __init__(self, output_dir: str | os.PathLike = "runtime/screenshots"):
        self.output_dir = Path(output_dir)

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        try:
            screenshots = sorted(
                self.output_dir.glob("*.png"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not screenshots:
                return ToolResult(
                    success=False,
                    tool=request.tool,
                    error="No screenshots have been captured yet.",
                    duration_seconds=time.perf_counter() - start,
                    metadata={"request_id": request.request_id},
                )

            path = screenshots[0].resolve()
            if platform.system() != "Windows":
                return ToolResult(
                    success=False,
                    tool=request.tool,
                    error="Opening screenshots automatically is currently supported on Windows only.",
                    duration_seconds=time.perf_counter() - start,
                    metadata={"request_id": request.request_id},
                )

            os.startfile(path)  # type: ignore[attr-defined]
            return ToolResult(
                success=True,
                tool=request.tool,
                output={"path": str(path), "opened": True},
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                error=f"Failed to open the latest screenshot: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )
