from __future__ import annotations

import json
import math
import re
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
    def _normalize_bbox(
        bbox: Any,
        *,
        width: int,
        height: int,
    ) -> dict[str, int]:
        """Convert normalized LFM grounding coordinates into screenshot pixels."""
        values: list[float]

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            values = [
                _number(bbox[0], "bbox[0]"),
                _number(bbox[1], "bbox[1]"),
                _number(bbox[2], "bbox[2]"),
                _number(bbox[3], "bbox[3]"),
            ]
            x1, y1, x2, y2 = values
        elif isinstance(bbox, dict):
            x1 = _number(bbox.get("x"), "bbox.x")
            y1 = _number(bbox.get("y"), "bbox.y")
            box_width = _number(bbox.get("width"), "bbox.width")
            box_height = _number(bbox.get("height"), "bbox.height")
            x2 = x1 + box_width
            y2 = y1 + box_height
            values = [x1, y1, x2, y2]
        else:
            raise ValueError("Vision model did not return a valid bounding box.")

        if any(value < 0.0 or value > 1000.0 for value in values):
            raise ValueError(
                "Vision model returned coordinates outside the normalized 0-1000 space."
            )
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Vision model returned a non-positive bounding box.")

        pixel_x1 = x1 / 1000.0 * width
        pixel_y1 = y1 / 1000.0 * height
        pixel_x2 = x2 / 1000.0 * width
        pixel_y2 = y2 / 1000.0 * height

        return {
            "x": int(round(pixel_x1)),
            "y": int(round(pixel_y1)),
            "width": max(1, int(round(pixel_x2 - pixel_x1))),
            "height": max(1, int(round(pixel_y2 - pixel_y1))),
        }

    @staticmethod
    def _recover_truncated_grounding(text: str) -> dict[str, Any] | None:
        """Recover critical grounding fields when JSON is cut off by token limits."""
        value = str(text or "").strip()
        found_match = re.search(
            r'"found"\s*:\s*(true|false)',
            value,
            flags=re.IGNORECASE,
        )
        element_match = re.search(
            r'"element"\s*:\s*"((?:\\.|[^"\\])*)"',
            value,
            flags=re.IGNORECASE,
        )
        bbox_match = re.search(
            r'"bbox"\s*:\s*\[\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*,\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*,\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*,\s*'
            r'([0-9]+(?:\.[0-9]+)?)',
            value,
            flags=re.IGNORECASE,
        )
        confidence_match = re.search(
            r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            value,
            flags=re.IGNORECASE,
        )

        if found_match is None or bbox_match is None or confidence_match is None:
            return None

        try:
            confidence = max(
                0.0,
                min(1.0, float(confidence_match.group(1))),
            )
            bbox = [float(bbox_match.group(index)) for index in range(1, 5)]
        except (TypeError, ValueError):
            return None

        element = ""
        if element_match is not None:
            try:
                element = json.loads(f'"{element_match.group(1)}"')
            except json.JSONDecodeError:
                element = element_match.group(1)

        return {
            "found": found_match.group(1).lower() == "true",
            "element": element,
            "bbox": bbox,
            "confidence": confidence,
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
            "Return its bounding box as normalized coordinates from 0 to 1000, "
            "with [x1, y1, x2, y2] measured from the screenshot top-left. "
            "Return found=false and bbox=null when absent or ambiguous."
        )
        system_prompt = (
            "You are A.S.T.A.'s visual grounding sensor. Inspect only the supplied image. "
            "Return exactly one compact JSON object with fields found, element, bbox, and confidence. "
            "When found=true, bbox must be [x1, y1, x2, y2] using normalized 0-1000 image coordinates "
            "(not desktop pixels). When found=false, bbox must be null. Only set found=true when the "
            "requested target is actually visible and unambiguous. Do not copy coordinates, labels, "
            "or conclusions from this prompt. Do not use prior context or assumed UI state. "
            "Return no markdown and no extra fields."
        )

        response_schema = {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "element": {"type": "string"},
                "bbox": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        {"type": "null"},
                    ],
                },
                "confidence": {"type": "number"},
            },
            "required": ["found", "element", "bbox", "confidence"],
            "additionalProperties": False,
        }

        try:
            inspection = self.vision_engine.inspect(
                image_path,
                prompt,
                json_mode=True,
                system_prompt=system_prompt,
                response_schema=response_schema,
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
        json_recovered = False
        if not isinstance(parsed, dict):
            recovered = self._recover_truncated_grounding(
                inspection.get("text", "")
            )
            if recovered is not None:
                parsed = recovered
                json_recovered = True
                print(
                    "[Vision] Locate: recovered truncated JSON "
                    f"target={target!r}",
                    flush=True,
                )
            else:
                print(
                    "[Vision] Locate failed: "
                    f"target={target!r} reason=invalid_json "
                    f"raw={str(inspection.get('text') or '')[:240]!r}",
                    flush=True,
                )
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
                    metadata={
                        "request_id": request.request_id,
                        "vision_model": inspection.get("model"),
                    },
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
            "target": target,
            "path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "found": found,
            "element": str(parsed.get("element") or target).strip(),
            "confidence": confidence,
            "json_recovered": json_recovered,
            "model": inspection.get("model"),
            "ttft": inspection.get("ttft"),
            "request_time": inspection.get("request_time"),
            "output_tokens": inspection.get("output_tokens"),
            "tokens_per_second": inspection.get("tokens_per_second"),
        }

        if not found:
            print(
                "[Vision] Locate failed: "
                f"target={target!r} reason=target_not_found "
                f"confidence={confidence:.2f} "
                f"json_recovered={str(json_recovered).lower()}",
                flush=True,
            )
            return ToolResult(
                success=False,
                tool=request.tool,
                output=base_output,
                error=f"Target not found: {target}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        try:
            bbox = self._normalize_bbox(
                parsed.get("bbox"),
                width=width,
                height=height,
            )
        except ValueError as exc:
            print(
                "[Vision] Locate failed: "
                f"target={target!r} reason=invalid_bbox error={str(exc)!r}",
                flush=True,
            )
            return ToolResult(
                success=False,
                tool=request.tool,
                output=base_output,
                error=f"Invalid visual grounding: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        if confidence < self.confidence_threshold:
            print(
                "[Vision] Locate failed: "
                f"target={target!r} reason=low_confidence "
                f"confidence={confidence:.2f} threshold={self.confidence_threshold:.2f}",
                flush=True,
            )
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
            f"center=({screen_center['x']},{screen_center['y']}) "
            f"json_recovered={str(json_recovered).lower()}",
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
