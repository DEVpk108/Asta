import os
import re
import time
from collections import Counter

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
        self.model_name = os.getenv("ASTA_STT_MODEL", str(model_name)).strip() or str(model_name)
        try:
            configured_beam_size = int(
                os.getenv("ASTA_STT_BEAM_SIZE", str(beam_size))
            )
        except ValueError:
            configured_beam_size = int(beam_size)
        self.beam_size = max(1, min(10, configured_beam_size))
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
                model_size_or_path=self.model_name,
                device=device,
                compute_type=compute_type,
            )
            print(
                f"[STT/Whisper] Ready "
                f"(model={self.model_name}, device={device}, "
                f"load={time.perf_counter() - start:.3f}s)",
                flush=True,
            )

        print(
            f"[STT] Backend: {self.backend} "
            f"(beam_size={self.beam_size})",
            flush=True,
        )

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
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower())
        return re.sub(r"\s+", " ", normalized).strip()


    def _is_hallucination(self, text):
        normalized = self._normalize(text)
        if not normalized:
            return True
        if normalized in self.HALLUCINATION_PHRASES:
            return True

        if normalized in {
            "okay thank you",
            "ok thank you",
            "okay thanks",
            "ok thanks",
        }:
            return True

        alnum = re.sub(r"[^a-z0-9]+", "", normalized)
        if not alnum:
            return True

        words = normalized.split()
        if len(words) >= 8 and len(set(words)) <= 2:
            return True

        return False


    @staticmethod
    def _segment_is_unreliable(segment, *, strict=False):
        """Reject segments that strongly look like silence/noise transcription."""
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        compression_ratio = float(
            getattr(segment, "compression_ratio", 0.0) or 0.0
        )

        if no_speech_prob >= 0.80 and avg_logprob <= -1.0:
            return True

        if strict and no_speech_prob >= 0.80:
            return True
        if compression_ratio >= 3.0 and avg_logprob <= -1.0:
            return True
        return False


    def _transcribe_whisper(self, audio, *, strict=False):
        if self.model is None:
            raise RuntimeError("Whisper backend is not initialized.")

        def decode(temperature):
            segments, info = self.model.transcribe(
                audio,
                language=getattr(self, "language", "en"),
                beam_size=getattr(self, "beam_size", 5),
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.80,
            )

            parts = []
            segment_stats = []
            rejected_segments = 0

            for segment in segments:
                if not segment.text or not segment.text.strip():
                    continue

                avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
                no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
                compression_ratio = float(
                    getattr(segment, "compression_ratio", 0.0) or 0.0
                )
                segment_stats.append(
                    (avg_logprob, no_speech_prob, compression_ratio)
                )

                if self._segment_is_unreliable(segment, strict=strict):
                    rejected_segments += 1
                    if not strict:
                        print(
                            "[STT] Rejected low-confidence segment: "
                            f"text={segment.text!r} "
                            f"avg_logprob={avg_logprob:.2f} "
                            f"no_speech={no_speech_prob:.2f} "
                            f"compression={compression_ratio:.2f}",
                            flush=True,
                        )
                    continue

                parts.append(segment.text.strip())

            language = getattr(info, "language", None)
            language_probability = float(
                getattr(info, "language_probability", 0.0) or 0.0
            )

            return (
                " ".join(parts).strip(),
                segment_stats,
                rejected_segments,
                language,
                language_probability,
            )

        text, segment_stats, rejected_segments, language, language_probability = decode(
            0.0
        )

        # A low-confidence first pass is commonly caused by a clipped onset
        # rather than a genuinely silent utterance. Retry the same captured
        # audio with a small amount of decoding temperature so Whisper can
        # escape a poor first-token choice without changing the normal fast
        # path for healthy utterances.
        if (
            text
            and segment_stats
            and not strict
            and (
                sum(item[1] for item in segment_stats) / len(segment_stats) >= 0.55
                or sum(item[0] for item in segment_stats) / len(segment_stats) <= -0.70
            )
        ):
            retry_text, retry_stats, retry_rejected, retry_language, retry_probability = decode(
                0.2
            )

            def quality(stats):
                if not stats:
                    return float("-inf")
                return sum(
                    avg + (1.0 - no_speech) * 0.5
                    for avg, no_speech, _compression in stats
                ) / len(stats)

            if retry_text and quality(retry_stats) > quality(segment_stats):
                print(
                    "[STT] Low-confidence retry selected "
                    f"(base={quality(segment_stats):.3f}, "
                    f"retry={quality(retry_stats):.3f}).",
                    flush=True,
                )
                text = retry_text
                segment_stats = retry_stats
                rejected_segments = retry_rejected
                language = retry_language
                language_probability = retry_probability
            else:
                print(
                    "[STT] Low-confidence retry did not improve the decode.",
                    flush=True,
                )

        self.last_language = language
        self.last_language_probability = language_probability

        if self.debug and segment_stats:
            formatted_stats = [
                f"avg_logprob={avg:.2f} no_speech={no_speech:.2f} compression={compression:.2f}"
                for avg, no_speech, compression in segment_stats
            ]
            counts = Counter(formatted_stats)
            unique_stats = []
            for stats in dict.fromkeys(formatted_stats):
                count = counts[stats]
                unique_stats.append(
                    f"{stats} (x{count})" if count > 1 else stats
                )

            print(
                f"[STT] Segment confidence ({len(segment_stats)} segment(s)): "
                + "; ".join(unique_stats),
                flush=True,
            )

        if rejected_segments and not text:
            print(
                "[STT] All Whisper segments rejected as unreliable.",
                flush=True,
            )

        return text

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

    def transcribe(self, audio, *, strict=False):
        if audio is None:
            return ""

        backend = getattr(self, "backend", "whisper")
        start = time.perf_counter()

        try:
            if backend == "indic":
                whisper_text = self._transcribe_whisper(audio)
                text = self._use_indic(audio, whisper_text)
            else:
                text = self._transcribe_whisper(audio, strict=strict)
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
