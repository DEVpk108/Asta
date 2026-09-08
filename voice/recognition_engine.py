from faster_whisper import WhisperModel
import torch


class RecognitionEngine:
    """Local speech-to-text engine for English, Hindi, and mixed speech.

    The previous configuration used ``medium.en`` and explicitly forced
    ``language="en"``. That made Hindi and Hinglish speech fundamentally
    difficult to recognize. A multilingual Whisper checkpoint with automatic
    language detection lets the recognizer choose between supported languages
    from the audio itself.
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

    def transcribe(self, audio):
        if audio is None:
            return ""

        try:
            segments, info = self.model.transcribe(
                audio,
                # None enables Whisper language detection. This is important
                # for English, Hindi, and mixed English/Hindi utterances.
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=(
                    "Conversation with ASTA. The speaker may use English, "
                    "Hindi, or natural Hinglish mixing both languages. "
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

        if self.debug:
            detected = getattr(info, "language", "unknown")
            probability = getattr(info, "language_probability", 0.0)
            print(f"[Voice] STT model: {self.model_name}", flush=True)
            print(f"[Language] {detected}", flush=True)
            print(f"[Probability] {probability:.2f}", flush=True)
            print(f"[Voice] {text}", flush=True)

        return text
