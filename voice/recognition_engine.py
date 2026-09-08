import os
import time

from faster_whisper import WhisperModel
import torch


class RecognitionEngine:
    """Pluggable local STT router.

    Backends:
      - whisper: use multilingual Faster-Whisper for every utterance.
      - indic: use AI4Bharat IndicConformer for an Indian language.
      - hybrid (default): Whisper performs language detection and English
        transcription; Hindi is re-decoded with IndicConformer.

    This keeps the existing English path intact while giving Hindi a native
    Indian-language ASR backend. IndicConformer is lazy-loaded only in hybrid
    mode after Hindi is detected, so A.S.T.A. does not pay its model startup
    cost until it is needed.
    """

    SUPPORTED_BACKENDS = {"whisper", "indic", "hybrid"}

    def __init__(
        self,
        model_name="medium",
        beam_size=5,
        language=None,
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
            (backend or os.getenv("ASTA_STT_BACKEND", "hybrid"))
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
        self.debug = True
        self.last_backend = None
        self.last_language = None
        self.last_language_probability = 0.0

        # Whisper remains the detector/English recognizer for hybrid mode. For
        # explicit Indic mode it is not loaded, keeping startup lighter.
        self.model = None
        if self.backend in {"whisper", "hybrid"}:
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

        print(
            f"[STT] Backend: {self.backend}",
            flush=True,
        )

    def _load_indic(self):
        if self._indic is None:
            from .indic_conformer_engine import IndicConformerEngine

            self._indic = IndicConformerEngine(
                model_id=self.indic_model_id,
                decoder=self.indic_decoder,
                language=self.indic_language,
            )
        return self._indic

    def _transcribe_whisper(self, audio):
        if self.model is None:
            raise RuntimeError("Whisper backend is not initialized.")

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # The outer Silero VAD already returns an utterance.
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=(
                "Conversation with ASTA. The speaker may use English, "
                "Hindi, or natural Hinglish. Preserve the spoken meaning "
                "and do not invent words."
            ),
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ).strip()

        self.last_language = getattr(info, "language", None)
        self.last_language_probability = float(
            getattr(info, "language_probability", 0.0) or 0.0
        )
        return text

    def transcribe(self, audio):
        if audio is None:
            return ""

        start = time.perf_counter()
        try:
            if self.backend == "indic":
                text = self._load_indic().transcribe(
                    audio,
                    language=self.indic_language,
                )
                self.last_backend = "indic-conformer"
                self.last_language = self.indic_language
                self.last_language_probability = 1.0
                return text

            # Whisper or hybrid starts with the same multilingual recognition
            # path. In hybrid mode a Hindi detection triggers a second decode
            # using IndicConformer.
            whisper_text = self._transcribe_whisper(audio)
            detected = (self.last_language or "").lower()

            if (
                self.backend == "hybrid"
                and detected == self.indic_language
                and self.last_language_probability >= 0.30
            ):
                indic = self._load_indic()
                indic_text = indic.transcribe(
                    audio,
                    language=self.indic_language,
                )
                if indic_text:
                    self.last_backend = "indic-conformer"
                    text = indic_text
                else:
                    self.last_backend = "whisper"
                    text = whisper_text
            else:
                self.last_backend = "whisper"
                text = whisper_text

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
