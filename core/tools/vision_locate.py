from __future__ import annotations

import math
import struct
import time
from pathlib import Path
from typing import Any, Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


def _png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            width, height = struct.unpack(">II", header[16:24])
            if width > 0 and height > 0:
                return width, height
    except OSError:
        pass
    return None


def _number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


class VisionLocateTool(Tool):
    """Locate a visual target on the current screen and ground it to coordinates."""

    def __init__(
        self,
        vision_engine,
        *,
        capture: Callable[[], str | Path | dict[str, Any]] | None = None,
        confidence_threshold: float = 0.70,
    ):
        self.vision_engine = vision_engine
        self.capture = capture
        self.confidence_threshold = max(
            0.0,
            min(1.0, float(confidence_threshold)),
        )

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vision.locate",
            description=(
                "Capture the current screen and locate a requested visual target, "
                "returning a grounded bounding box and screen coordinates."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The button, control, icon, field, window, or other visual target to locate.",
                    },
                },
                "required": ["target"],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["locate", "find_visual_target", "ground_target"],
                "action_aliases": ["find", "locate_on_screen", "ground"],
                "category": "vision",
            },
        )

    @staticmethod
    def _capture_metadata(
        captured: str | Path | dict[str, Any],
    ) -> tuple[Path, int, int, int, int]:
        origin_x = 0
        origin_y = 0
        width = 0
        height = 0

        if isinstance(captured, dict):
            captured_path = captured.get("path")
            if not captured_path:
                raise ValueError(
                    "Screenshot capture returned metadata without a 'path'."
                )
            image_path = Path(str(captured_path))
            width = int(captured.get("width") or 0)
            height = int(captured.get("height") or 0)
            origin_x = int(captured.get("origin_x") or 0)
            origin_y = int(captured.get("origin_y") or 0)
        elif isinstance(captured, (str, Path)):
            image_path = Path(captured)
        else:
            raise TypeError(
                "Screenshot capture must return a path or metadata dict containing 'path'."
            )

        if width <= 0 or height <= 0:
            size = _png_size(image_path)
            if size is not None:
                width, height = size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Screenshot dimensions are unavailable; capture metadata must include width and height."
            )

        if not image_path.is_file():
            raise ValueError(f"Screenshot image does not exist: {image_path}")

        return image_path, width, height, origin_x, origin_y

    @staticmethod
    def _validate_bbox(
        bbox: Any,
        *,
        width: int,
        height: int,
    ) -> dict[str, int]:
        if not isinstance(bbox, dict):
            raise ValueError("Vision model did not return a bounding box.")

        x = _number(bbox.get("x"), "bbox.x")
        y = _number(bbox.get("y"), "bbox.y")
        box_width = _number(bbox.get("width"), "bbox.width")
        box_height = _number(bbox.get("height"), "bbox.height")

        if box_width <= 0 or box_height <= 0:
            raise ValueError("Vision model returned a non-positive bounding box.")

        right = x + box_width
        bottom = y + box_height
        if x < 0 or y < 0 or right > width or bottom > height:
            raise ValueError(
                "Vision model returned a bounding box outside the screenshot bounds."
            )

        return {
            "x": int(round(x)),
            "y": int(round(y)),
            "width": int(round(box_width)),
            "height": int(round(box_height)),
        }

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = str(request.arguments.get("target") or "").strip()
        if not target:
            return ToolResult(
                success=False,
                tool=request.tool,
                error="Argument 'target' must be a non-empty string.",
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
            captured = self.capture()
            image_path, width, height, origin_x, origin_y = self._capture_metadata(captured)
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                error=f"Screenshot capture failed: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        prompt = (
            f"Locate the visual target '{target}' in the supplied screenshot. "
            "Use absolute pixel coordinates relative to the screenshot's top-left corner. "
            "Return the smallest bounding box around the target. "
            "If the target is absent or ambiguous, set found=false and use null bbox."
        )
        system_prompt = (
            "You are A.S.T.A.'s visual grounding sensor. Inspect only the supplied image. "
            "Return exactly one compact JSON object with these fields: "
            '{"found":true,"element":"target label","bbox":{"x":100,"y":200,"width":120,"height":40},"confidence":0.90,"summary":"short reason"}. '
            "When not found or ambiguous, return found=false and bbox=null. "
            "Coordinates must be image pixels from the screenshot top-left. "
            "Do not return markdown. Keep summary <= 10 words."
        )

        try:
            inspection = self.vision_engine.inspect(
                image_path,
                prompt,
                json_mode=True,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                output={"path": str(image_path.resolve())},
                error=f"Vision grounding failed: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        parsed = inspection.get("json")
        if not isinstance(parsed, dict):
            return ToolResult(
                success=False,
                tool=request.tool,
                output={
                    "path": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                    "origin_x": origin_x,
                    "origin_y": origin_y,
                    "raw_response": inspection.get("text", ""),
                },
                error="Vision grounding returned invalid JSON.",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        found = bool(parsed.get("found", False))
        try:
            confidence = max(
                0.0,
                min(1.0, float(parsed.get("confidence", 0.0))),
            )
        except (TypeError, ValueError):
            confidence = 0.0

        base_output = {
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "found": found,
            "element": str(parsed.get("element") or target).strip(),
            "confidence": confidence,
            "summary": str(parsed.get("summary") or "").strip(),
            "model": inspection.get("model"),
            "ttft": inspection.get("ttft"),
            "request_time": inspection.get("request_time"),
            "output_tokens": inspection.get("output_tokens"),
            "tokens_per_second": inspection.get("tokens_per_second"),
        }

        if not found:
            return ToolResult(
                success=False,
                tool=request.tool,
                output=base_output,
                error=f"Target not found: {target}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        try:
            bbox = self._validate_bbox(
                parsed.get("bbox"),
                width=width,
                height=height,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                tool=request.tool,
                output=base_output,
                error=f"Invalid visual grounding: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        if confidence < self.confidence_threshold:
            return ToolResult(
                success=False,
                tool=request.tool,
                output={
                    **base_output,
                    "bbox": bbox,
                },
                error=(
                    f"Visual grounding confidence {confidence:.2f} is below "
                    f"the required threshold {self.confidence_threshold:.2f}."
                ),
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        center = {
            "x": int(round(bbox["x"] + bbox["width"] / 2)),
            "y": int(round(bbox["y"] + bbox["height"] / 2)),
        }
        screen_bbox = {
            **bbox,
            "x": bbox["x"] + origin_x,
            "y": bbox["y"] + origin_y,
        }
        screen_center = {
            "x": center["x"] + origin_x,
            "y": center["y"] + origin_y,
        }

        output = {
            **base_output,
            "bbox": bbox,
            "center": center,
            "screen_bbox": screen_bbox,
            "screen_center": screen_center,
        }

        print(
            "[Vision] Locate: "
            f"target={target!r} "
            f"found=true "
            f"confidence={confidence:.2f} "
            f"center=({screen_center['x']},{screen_center['y']})",
            flush=True,
        )

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
