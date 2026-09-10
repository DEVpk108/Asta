import os
import time

import numpy as np
import torch


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerUnavailable(RuntimeError):
    """Raised when the optional IndicConformer backend cannot be loaded."""


class IndicConformerEngine:
    """AI4Bharat IndicConformer backend for Indian-language ASR.

    The backend is optional. Hugging Face authentication is read from the
    environment and is never stored in the repository.
    """

    def __init__(self, model_id=MODEL_ID, decoder="rnnt", language="hi"):
        self.model_id = model_id
        self.decoder = decoder
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.last_duration = 0.0
        self._load_error = None
        self._load()

    def _load(self):
        start = time.perf_counter()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            self._load_error = exc
            raise IndicConformerUnavailable(
                "IndicConformer requires transformers and its audio dependencies. "
                "Install voice/requirements-indic.txt."
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        kwargs = {"trust_remote_code": True}
        if token:
            kwargs["token"] = token

        try:
            self.model = AutoModel.from_pretrained(self.model_id, **kwargs)
        except Exception as exc:
            self._load_error = exc
            message = str(exc)
            if "gated repo" in message.lower() or "401 client error" in message.lower():
                message = (
                    "IndicConformer is gated on Hugging Face. Accept the model terms "
                    "and set HF_TOKEN or HUGGINGFACE_HUB_TOKEN locally."
                )
            raise IndicConformerUnavailable(message) from exc

        self.model = self.model.to(self.device)
        self.model.eval()
        print(
            f"[STT/IndicConformer] Ready "
            f"(model={self.model_id}, decoder={self.decoder}, "
            f"language={self.language}, device={self.device}, "
            f"load={time.perf_counter() - start:.3f}s)",
            flush=True,
        )

    def transcribe(self, audio, language=None):
        if audio is None:
            return ""

        target_language = language or self.language
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return ""

        tensor = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0)
        if self.device == "cuda":
            tensor = tensor.to(device="cuda", non_blocking=True)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                result = self.model(tensor, target_language, self.decoder)
        except TypeError:
            with torch.inference_mode():
                result = self.model(
                    tensor,
                    target_language,
                    decoding_method=self.decoder,
                )

        self.last_duration = time.perf_counter() - start
        text = self._extract_text(result)
        print(
            f"[STT/IndicConformer] {target_language}/{self.decoder} "
            f"{self.last_duration:.3f}s: {text}",
            flush=True,
        )
        return text

    @staticmethod
    def _extract_text(result):
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, (list, tuple)):
            parts = []
            for item in result:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, (list, tuple)):
                    parts.extend(
                        str(part).strip()
                        for part in item
                        if str(part).strip()
                    )
            return " ".join(parts).strip()
        return str(result).strip()
