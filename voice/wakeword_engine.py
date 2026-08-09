from collections import deque
from pathlib import Path

import numpy as np
from openwakeword.model import Model

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = (
    ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "models"
    / "hello_asta.onnx"
)

class WakeWordEngine:

    def __init__(
        self,
        model_path=None,
        threshold=0.5,
        debug=True,
    ):
        self.model_path = str(model_path or DEFAULT_MODEL)
        self.threshold = threshold
        self.debug = debug

        self.model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
            vad_threshold=0,
        )

        self.prediction_history = deque(maxlen=5)

        if "hello_asta" not in self.model.models:
            raise RuntimeError(
                f"hello_asta model was not loaded. "
                f"Loaded models: {list(self.model.models.keys())}"
            )

        print(f"[WakeWord] Loaded: {model_path}")
        print(f"[WakeWord] Threshold: {self.threshold}")

    def wait_for_wakeword(self, microphone):

        print("[WakeWord] Listening for hello_asta...")

        buffer = deque()

        while True:

            chunk = microphone.get_chunk().flatten()

            buffer.extend(chunk)

            if len(buffer) < 1280:
                continue

            audio = np.array(
                [buffer.popleft() for _ in range(1280)],
                dtype=np.float32,
            )

            # Microphone is expected to provide normalized float audio.
            audio_int16 = np.clip(
                audio * 32767,
                -32768,
                32767,
            ).astype(np.int16)

            prediction = self.model.predict(audio_int16)

            score = float(prediction["hello_asta"])

            if self.debug:
                print(
                    f"\r[WakeWord] hello_asta: {score:.3f}",
                    end="",
                    flush=True,
                )

            if score >= self.threshold:
                print(
                    f"\n[WakeWord] hello_asta detected "
                    f"(score={score:.3f})"
                )

                return microphone.get_buffer()