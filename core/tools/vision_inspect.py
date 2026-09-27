from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


class VisionInspectTool(Tool):
    """Capture the current screen and inspect it with A.S.T.A.'s visual model."""

    def __init__(
        self,
        vision_engine,
        *,
        capture: Callable[[], str | Path] | None = None,
    ):
        self.vision_engine = vision_engine
        self.capture = capture

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.inspect",
            description=(
                "Capture the current screen and ask the configured vision model "
                "a focused visual question."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["inspect", "visual_verify"],
                "action_aliases": ["verify_screen", "inspect_screen"],
                "category": "vision",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        prompt = str(request.arguments.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(
                success=False,
                tool=request.tool,
                error="Argument 'prompt' must be a non-empty string.",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        if self.capture is None:
            return ToolResult(
                success=False,
                tool=request.tool,
                error="Screenshot capture is not configured.",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        try:
            image_path = Path(self.capture())
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                error=f"Screenshot capture failed: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        try:
            inspection = self.vision_engine.inspect(
                image_path,
                prompt,
                json_mode=True,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                output={"path": str(image_path.resolve())},
                error=f"Vision inspection failed: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        parsed = inspection.get("json") or {}
        if not isinstance(parsed, dict):
            parsed = {}

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        visual_match = bool(parsed.get("visual_match", False))
        verified = bool(
            visual_match and confidence >= 0.70
        )

        output = {
            "path": str(image_path.resolve()),
            "analysis": parsed,
            "summary": str(parsed.get("summary") or inspection.get("text") or ""),
            "visual_match": visual_match,
            "verified": verified,
            "confidence": confidence,
            "model": inspection.get("model"),
            "ttft": inspection.get("ttft"),
            "request_time": inspection.get("request_time"),
            "output_tokens": inspection.get("output_tokens"),
            "tokens_per_second": inspection.get("tokens_per_second"),
        }

        return ToolResult(
            success=True,
            tool=request.tool,
            output=output,
            duration_seconds=time.perf_counter() - start,
            metadata={
                "request_id": request.request_id,
                "vision_model": inspection.get("model"),
            },
        )
