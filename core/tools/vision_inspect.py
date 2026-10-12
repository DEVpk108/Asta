from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


def _extract_required_playback_text(prompt: str) -> str | None:
    """Extract an explicitly requested playback title from the verifier prompt."""
    match = re.search(
        r"Verify that ['\"](.+?)['\"] is actually playing",
        str(prompt or ""),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _text_contains_required_phrase(text: str, required: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    wanted = normalize(required)
    haystack = normalize(text)
    if not wanted or not haystack:
        return False
    return wanted in haystack


def _extract_playback_app(prompt: str) -> str:
    match = re.search(
        r"is actually playing in (.+?)\.(?:\s|$)",
        str(prompt or ""),
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _default_media_session_reader():
    import os
    import platform

    if platform.system() != "Windows":
        return None
    if os.getenv("ASTA_MEDIA_SESSION", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    from core.media.smtc import media_session

    return media_session


def _format_seconds(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "n/a"


class VisionInspectTool(Tool):
    """Capture the current screen and inspect it with A.S.T.A.'s visual model."""

    def __init__(
        self,
        vision_engine,
        *,
        capture: Callable[[], str | Path] | None = None,
        media_session_reader: Callable[..., dict] | None = None,
    ):
        self.vision_engine = vision_engine
        self.capture = capture
        # Reads the OS "now playing" session (Windows SMTC). Injected in tests.
        self.media_session_reader = (
            media_session_reader
            if media_session_reader is not None
            else _default_media_session_reader()
        )

    def _media_session_playback(self, title: str, app: str) -> dict | None:
        """Confirm playback from the OS media session, if it clearly matches."""
        reader = self.media_session_reader
        if reader is None or not title:
            return None
        hint = app.lower().replace(" ", "").replace("_", "")
        for attempt in range(2):
            try:
                info = reader("", hint) or {}
            except Exception:
                info = {}
            session_title = str(info.get("title") or "")
            playing = str(info.get("status") or "").strip().lower() == "playing"
            if playing and (
                _text_contains_required_phrase(session_title, title)
                or _text_contains_required_phrase(title, session_title)
            ):
                return info
            if attempt == 0:
                time.sleep(1.0)
        return None

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

        required_title = _extract_required_playback_text(prompt)
        if required_title:
            # Ask the OS what is playing first: exact, ~1s, and it avoids a slow
            # vision call that often cannot read the player bar reliably.
            session = self._media_session_playback(
                required_title, _extract_playback_app(prompt)
            )
            if session is not None:
                artist = str(session.get("artist") or "").strip()
                track = str(session.get("title") or required_title)
                track = f"{track} by {artist}" if artist else track
                print(f"[Vision] Inspect: media session confirms '{track}' is playing.", flush=True)
                return ToolResult(
                    success=True,
                    tool=request.tool,
                    output={
                        "summary": f"{track} is playing.",
                        "visual_match": True,
                        "verified": True,
                        "confidence": 1.0,
                        "source": "media_session",
                        "track": track,
                        "session_app": session.get("app"),
                        "required_playback_text": required_title,
                        "question": False,
                    },
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
            if isinstance(captured, dict):
                captured_path = captured.get("path")
                if not captured_path:
                    raise ValueError(
                        "Screenshot capture returned metadata without a 'path'."
                    )
                image_path = Path(str(captured_path))
            elif isinstance(captured, (str, Path)):
                image_path = Path(captured)
            else:
                raise TypeError(
                    "Screenshot capture must return a path or metadata dict containing 'path'."
                )
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
        verification_guard = None
        required_playback_text = _extract_required_playback_text(prompt)
        evidence_text = " ".join(
            [
                str(parsed.get("summary") or ""),
                str(parsed.get("observations") or ""),
                str(inspection.get("text") or ""),
            ]
        )
        if (
            visual_match
            and required_playback_text
            and not _text_contains_required_phrase(
                evidence_text,
                required_playback_text,
            )
        ):
            visual_match = False
            verification_guard = (
                "Model evidence did not contain the explicitly requested "
                f"playback title '{required_playback_text}'."
            )

        verified = bool(
            visual_match and confidence >= 0.70
        )

        output = {
            "path": str(image_path.resolve()),
            "analysis": parsed,
            "summary": str(parsed.get("summary") or inspection.get("text") or ""),
            "visual_match": visual_match,
            "verified": verified,
            "question": str(prompt or "").startswith(
                "Answer the user's question about the current screenshot"
            ),
            "confidence": confidence,
            "required_playback_text": required_playback_text,
            "verification_guard": verification_guard,
            "model": inspection.get("model"),
            "ttft": inspection.get("ttft"),
            "request_time": inspection.get("request_time"),
            "output_tokens": inspection.get("output_tokens"),
            "tokens_per_second": inspection.get("tokens_per_second"),
        }

        print(
            "[Vision] Inspect: "
            f"model={inspection.get('model') or 'unknown'} "
            f"ttft={_format_seconds(inspection.get('ttft'))} "
            f"request={_format_seconds(inspection.get('request_time'))} "
            f"output={int(inspection.get('output_tokens') or 0)} tok "
            f"speed={float(inspection.get('tokens_per_second') or 0.0):.2f} tok/s "
            f"visual_match={str(visual_match).lower()} "
            f"confidence={confidence:.2f} "
            f"verified={str(verified).lower()} "
            f"guarded={str(bool(verification_guard)).lower()} "
            f"json_recovered={str(bool(inspection.get('json_recovered'))).lower()}",
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
