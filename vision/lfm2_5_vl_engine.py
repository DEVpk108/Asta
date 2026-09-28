from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .vision_server_manager import VisionServerManager


class VisionEngineError(RuntimeError):
    """Raised when the configured multimodal vision backend cannot respond."""


class LFM25VLEngine:
    """OpenAI-compatible client for a local LFM2.5-VL-1.6B server.

    The engine is intentionally independent from A.S.T.A.'s main text LLM.
    It can point at a second llama.cpp/LM Studio/vLLM-compatible endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        session: requests.Session | None = None,
        server_manager: VisionServerManager | None = None,
    ):
        self.base_url = (
            base_url
            or os.getenv(
                "ASTA_VISION_BASE_URL",
                "http://127.0.0.1:8090/v1",
            )
        ).rstrip("/")
        self.model = model or os.getenv("ASTA_VISION_MODEL") or None
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("ASTA_VISION_TIMEOUT", "60")
        )
        self.max_output_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else os.getenv("ASTA_VISION_MAX_OUTPUT_TOKENS", "64")
        )
        self.temperature = float(
            temperature
            if temperature is not None
            else os.getenv("ASTA_VISION_TEMPERATURE", "0")
        )
        self.session = session or requests.Session()
        self.server_manager = server_manager or VisionServerManager(base_url=self.base_url)
        self.models_url = f"{self.base_url}/models"
        self.chat_url = f"{self.base_url}/chat/completions"

        host = self.base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        if host in {"localhost", "127.0.0.1", "::1"}:
            self.session.trust_env = False

    def _ensure_server(self) -> None:
        if not self.server_manager.ensure_running():
            raise VisionEngineError(
                "The local LFM2.5-VL server could not be started or reached. "
                "Check ASTA_VISION_BASE_URL and the vision model configuration."
            )

    def _discover_model(self) -> str:
        self._ensure_server()
        try:
            response = self.session.get(
                self.models_url,
                timeout=self.timeout,
            )
            response.raise_for_status()
            models = response.json().get("data") or []
        except requests.RequestException as exc:
            raise VisionEngineError(
                f"Vision server is unavailable at {self.models_url}: {exc}"
            ) from exc

        if not models:
            raise VisionEngineError("Vision server returned no models.")

        if self.model:
            ids = {str(item.get("id") or "") for item in models}
            if self.model not in ids:
                raise VisionEngineError(
                    f"Configured vision model '{self.model}' is not exposed by the server."
                )
            return self.model

        model_id = str(models[0].get("id") or "").strip()
        if not model_id:
            raise VisionEngineError("Vision server returned a model without an id.")

        self.model = model_id
        return model_id

    @staticmethod
    def _image_data_url(path: str | os.PathLike[str]) -> str:
        image_path = Path(path)
        if not image_path.is_file():
            raise VisionEngineError(f"Vision image does not exist: {image_path}")

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        value = str(text or "").strip()
        candidates = [value]
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            candidates.append(value[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _recover_truncated_json(text: str) -> dict[str, Any] | None:
        """Recover the critical verification fields from a token-truncated JSON response."""
        value = str(text or "").strip()
        visual_match = re.search(
            r'"visual_match"\s*:\s*(true|false)',
            value,
            flags=re.IGNORECASE,
        )
        confidence_match = re.search(
            r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            value,
            flags=re.IGNORECASE,
        )
        summary_match = re.search(
            r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"',
            value,
            flags=re.IGNORECASE,
        )

        if visual_match is None or confidence_match is None:
            return None

        try:
            confidence = max(
                0.0,
                min(1.0, float(confidence_match.group(1))),
            )
        except ValueError:
            return None

        summary = ""
        if summary_match is not None:
            try:
                summary = json.loads('"{}"'.format(summary_match.group(1)))
            except json.JSONDecodeError:
                summary = summary_match.group(1).replace('\\"', '"')

        return {
            "visual_match": visual_match.group(1).lower() == "true",
            "confidence": confidence,
            "summary": summary,
        }

    def warmup(self) -> bool:
        try:
            self._discover_model()
            return True
        except VisionEngineError as exc:
            print(f"[Vision] Warm-up unavailable: {exc}", flush=True)
            return False

    def shutdown(self) -> None:
        self.server_manager.stop()

    def inspect(
        self,
        image_path: str | os.PathLike[str],
        prompt: str,
        *,
        json_mode: bool = True,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        model = self._discover_model()
        data_url = self._image_data_url(image_path)

        if json_mode:
            system = system_prompt or (
                "You are A.S.T.A.'s fast visual sensor. Inspect only the supplied image "
                "and the current user request. Do not rely on prior turns or assumed UI state. "
                "Return ONLY one compact JSON object with exactly these fields: "
                '{"visual_match":false,"confidence":0.0,"summary":"The requested condition is not clearly visible."}. '
                "Set visual_match=true only when the supplied image clearly supports the "
                "current request. If the image shows a different application, a different "
                "target, or insufficient evidence, set visual_match=false. Ignore A.S.T.A.'s "
                "HUD, conversation panels, assistant messages, terminal output, subtitles, "
                "and overlays as evidence unless the request explicitly asks about them. "
                "Do not treat text that merely repeats the requested command as proof that "
                "the underlying UI state exists. Never copy a target name or conclusion from "
                "examples or prior context. No markdown, no observations array, no extra "
                "keys, summary <= 12 words."
            )
        else:
            system = system_prompt or (
                "You are A.S.T.A.'s fast visual perception module. Answer only from the "
                "supplied image and do not invent details."
            )

        content = [
            {"type": "text", "text": str(prompt).strip()},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "timings_per_token": True,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "cache_prompt": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request_start = time.perf_counter()
        first_delta_time = None
        full_text = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        response_id = None

        try:
            with self.session.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(
                    chunk_size=1,
                    decode_unicode=True,
                ):
                    if not raw_line or not raw_line.startswith("data:"):
                        continue

                    raw = raw_line[len("data:") :].strip()
                    if raw == "[DONE]":
                        continue

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if event.get("error"):
                        error = event.get("error")
                        raise VisionEngineError(
                            f"Vision server returned an error: {error}"
                        )

                    response_id = event.get("id") or response_id
                    if event.get("usage"):
                        usage = dict(event["usage"])
                    if event.get("timings"):
                        timings = dict(event["timings"])

                    choices = event.get("choices") or []
                    choice = choices[0] if choices else {}
                    delta = choice.get("delta") or {}

                    text = delta.get("content") or ""
                    if text:
                        if first_delta_time is None:
                            first_delta_time = time.perf_counter()
                        full_text.append(str(text))
        except requests.RequestException as exc:
            raise VisionEngineError(
                f"Vision inference request failed: {type(exc).__name__}: {exc}"
            ) from exc

        text = "".join(full_text).strip()
        elapsed = time.perf_counter() - request_start
        ttft = (
            first_delta_time - request_start
            if first_delta_time is not None
            else None
        )
        output_tokens = int(
            timings.get("predicted_n")
            or usage.get("completion_tokens")
            or 0
        )
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
        tokens_per_second = float(
            timings.get("predicted_per_second")
            or (
                output_tokens / (predicted_ms / 1000.0)
                if output_tokens and predicted_ms
                else 0.0
            )
        )

        parsed = self._parse_json(text) if json_mode else None
        recovered = False
        if json_mode and parsed is None:
            parsed = self._recover_truncated_json(text)
            recovered = parsed is not None

        result: dict[str, Any] = {
            "model": model,
            "text": text,
            "json": parsed,
            "json_recovered": recovered,
            "image_path": str(Path(image_path).resolve()),
            "ttft": ttft,
            "request_time": elapsed,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "timings": dict(timings),
            "usage": dict(usage),
            "id": response_id,
        }

        if json_mode and parsed is None:
            result["verified"] = False
            result["visual_match"] = False

        return result
