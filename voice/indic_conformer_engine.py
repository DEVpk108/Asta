import time

import numpy as np
import torch


MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicConformerEngine:
    """AI4Bharat IndicConformer backend for Indian-language ASR.

    The official multilingual checkpoint is a 600M-parameter hybrid
    CTC/RNNT model covering 22 scheduled Indian languages, including Hindi.
    A.S.T.A. supplies already-segmented 16 kHz mono float32 audio.
    """

    def __init__(self, model_id=MODEL_ID, decoder="rnnt", language="hi"):
        self.model_id = model_id
        self.decoder = decoder
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.last_duration = 0.0

        self._load()

    def _load(self):
        start = time.perf_counter()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "IndicConformer requires transformers. Install it with: "
                "pip install transformers torchaudio onnx onnxruntime onnxruntime-gpu"
            ) from exc

        try:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device=self.device,
            )
        except TypeError:
            # Some released remote-code revisions do not expose the device
            # keyword even though the model itself can be moved to a device.
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
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
            tensor = tensor.cuda(non_blocking=True)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                result = self.model(
                    tensor,
                    target_language,
                    self.decoder,
                )
        except TypeError:
            # Keep the backend compatible with remote-code revisions that use
            # keyword arguments for the decoding mode.
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
