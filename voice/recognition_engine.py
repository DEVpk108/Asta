import os
import re
import time

from faster_whisper import WhisperModel
import torch


class RecognitionEngine:
    """Pluggable local STT router with Whisper as the current default."""

    SUPPORTED_BACKENDS = {"whisper", "indic", "hybrid"}

    # Phrases Whisper may invent from silence/noise. Never pass them to the AI.
    HALLUCINATION_PHRASES = {
        "the speaker is using english",
        "the speaker is speaking english",
        "the speaker speaks english",
        "thank you for watching",
        "thanks for watching",
        "subscribe",
        "please subscribe",
        "you",
        "thank you",
    }

    def __init__(
        self,
        model_name="medium",
        beam_size=5,
        language="en",
        backend=None,
        indic_model_id="ai4bharat/indic-conformer-600m-multilingual",
        indic_decoder="rnnt",
        indic_language="hi",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.device = device
        self.model_name = model_name
        self.beam_size = beam_size
        self.language = language
        self.backend = (
            (backend or os.getenv("ASTA_STT_BACKEND", "whisper"))
            .strip()
            .lower()
        )
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported STT backend '{self.backend}'. "
                f"Choose one of: {', '.join(sorted(self.SUPPORTED_BACKENDS))}."
            )

        self.indic_model_id = indic_model_id
        self.indic_decoder = indic_decoder
        self.indic_language = indic_language
        self._indic = None
        self._indic_unavailable = False
        self._indic_error = None
        self.debug = True
        self.last_backend = None
        self.last_language = None
        self.last_language_probability = 0.0

        self.model = None
        if self.backend in {"whisper", "hybrid", "indic"}:
            start = time.perf_counter()
            self.model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
            )
            print(
                f"[STT/Whisper] Ready "
                f"(model={model_name}, device={device}, "
                f"load={time.perf_counter() - start:.3f}s)",
                flush=True,
            )

        print(f"[STT] Backend: {self.backend}", flush=True)

    def _load_indic(self):
        if self._indic is not None:
            return self._indic
        if self._indic_unavailable:
            return None

        try:
            from .indic_conformer_engine import IndicConformerEngine

            self._indic = IndicConformerEngine(
                model_id=self.indic_model_id,
                decoder=self.indic_decoder,
                language=self.indic_language,
            )
            return self._indic
        except Exception as exc:
            self._indic_unavailable = True
            self._indic_error = exc
            print(
                f"[STT/IndicConformer] Unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print("[STT] Falling back to Whisper.", flush=True)
            return None

    @staticmethod
    def _normalize(text):
        return re.sub(r"\s+", " ", text.strip().lower())

    def _is_hallucination(self, text):
        normalized = self._normalize(text)
        if not normalized:
            return True
        if normalized in self.HALLUCINATION_PHRASES:
            return True

        alnum = re.sub(r"[^a-z0-9]+", "", normalized)
        if not alnum:
            return True

        words = normalized.split()
        if len(words) >= 8 and len(set(words)) <= 2:
            return True

        return False

    def _transcribe_whisper(self, audio):
        if self.model is None:
            raise RuntimeError("Whisper backend is not initialized.")

        # A.S.T.A. already extracts speech with its streaming Silero VAD.
        # Running another VAD pass here can clip/drop valid speech, so Whisper
        # receives the already-segmented command directly.
        segments, info = self.model.transcribe(
            audio,
            language=getattr(self, "language", "en"),
            beam_size=getattr(self, "beam_size", 5),
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.80,
        )

        parts = []
        segment_stats = []
        for segment in segments:
            if not segment.text or not segment.text.strip():
                continue
            parts.append(segment.text.strip())
            segment_stats.append(
                (
                    float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                    float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                    float(getattr(segment, "compression_ratio", 0.0) or 0.0),
                )
            )

        self.last_language = getattr(info, "language", None)
        self.last_language_probability = float(
            getattr(info, "language_probability", 0.0) or 0.0
        )

        if self.debug and segment_stats:
            stats_text = ", ".join(
                f"avg_logprob={avg:.2f} no_speech={no_speech:.2f} compression={compression:.2f}"
                for avg, no_speech, compression in segment_stats
            )
            print(f"[STT] Segment confidence: {stats_text}", flush=True)

        return " ".join(parts).strip()

    def _use_indic(self, audio, whisper_text):
        indic = self._load_indic()
        if indic is None:
            self.last_backend = "whisper-fallback"
            return whisper_text

        try:
            indic_text = indic.transcribe(
                audio,
                language=self.indic_language,
            )
        except Exception as exc:
            self._indic_unavailable = True
            self._indic_error = exc
            self.last_backend = "whisper-fallback"
            print(
                f"[STT/IndicConformer] Inference unavailable: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print("[STT] Falling back to Whisper.", flush=True)
            return whisper_text

        if indic_text:
            self.last_backend = "indic-conformer"
            return indic_text

        self.last_backend = "whisper-fallback"
        return whisper_text

    def transcribe(self, audio):
        if audio is None:
            return ""

        backend = getattr(self, "backend", "whisper")
        start = time.perf_counter()

        try:
            if backend == "indic":
                whisper_text = self._transcribe_whisper(audio)
                text = self._use_indic(audio, whisper_text)
            else:
                text = self._transcribe_whisper(audio)
                self.last_backend = "whisper"

            if self._is_hallucination(text):
                print(f"[STT] Rejected likely hallucination: {text!r}", flush=True)
                return ""

            elapsed = time.perf_counter() - start
            print(
                f"[STT] {self.last_backend} "
                f"language={self.last_language} "
                f"prob={self.last_language_probability:.2f} "
                f"time={elapsed:.3f}s: {text}",
                flush=True,
            )
            return text
        except Exception as exc:
            print(
                f"[Voice] Recognition error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""
