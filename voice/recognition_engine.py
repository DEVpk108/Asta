from faster_whisper import WhisperModel
import torch

class RecognitionEngine:
    def __init__(self,model_name="medium.en",):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        compute_type = (
            "float16"
            if device == "cuda"
            else "int8"
        )
        
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
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt="The following is a conversation with ASTA.",
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()
        except Exception as e:
            print(f"[Voice] {e}")
            return ""
        if self.debug:
            print(f"[Language] {info.language}")
            print(f"[Probability] {info.language_probability:.2f}")
            print("[Voice]", text)

        return text