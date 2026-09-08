from faster_whisper import WhisperModel
import torch


class RecognitionEngine:
    """Local speech-to-text engine for English, Hindi, and mixed speech.

    The recognizer uses a multilingual Whisper checkpoint with automatic
    language detection. Audio is already segmented by A.S.T.A.'s outer
    Silero VAD, so Whisper's second VAD pass is intentionally disabled to
    reduce the chance of chopping short Hindi/Hinglish phrases.
    """

    def __init__(
        self,
        model_name="medium",
        beam_size=5,
        language=None,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.model_name = model_name
        self.beam_size = beam_size
        self.language = language
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type=compute_type,
        )
        self.debug = False
        self.last_language = None
        self.last_language_probability = 0.0

    def transcribe(self, audio):
        if audio is None:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio,
                # None enables Whisper language detection for English, Hindi,
                # and mixed English/Hindi utterances.
                language=self.language,
                beam_size=self.beam_size,
                # The outer Silero VAD already returns an utterance. Running
                # another VAD here can discard weak/short syllables.
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                initial_prompt=(
                    "Conversation with ASTA. The speaker may use English, "
                    "Hindi, or natural Hinglish. Common Hindi words may be "
                    "spoken in Roman script, for example namaste, kya, "
                    "kaise, haal, hai, mujhe, tumhe, aap, mera, meri. "
                    "Preserve the spoken meaning and do not invent words."
                ),
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text and segment.text.strip()
            ).strip()
        except Exception as exc:
            print(f"[Voice] Recognition error: {type(exc).__name__}: {exc}", flush=True)
            return ""

        self.last_language = getattr(info, "language", None)
        self.last_language_probability = float(
            getattr(info, "language_probability", 0.0) or 0.0
        )

        if self.debug:
            print(f"[Voice] STT model: {self.model_name}", flush=True)
            print(
                f"[Language] {self.last_language} "
                f"(prob={self.last_language_probability:.2f})",
                flush=True,
            )
            print(f"[Voice] {text}", flush=True)

        return text
