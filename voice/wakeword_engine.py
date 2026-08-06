from collections import deque

import numpy as np
from ai.wakeword.openwakeword.model import Model


class WakeWordEngine:

    def __init__(
        self,
        model_path="generated/models/asta.onnx",
        threshold=0.5,
        debug=True,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.debug = debug

        self.model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
            vad_threshold=0,
        )

        if "asta" not in self.model.models:
            raise RuntimeError(
                f"ASTA model was not loaded. "
                f"Loaded models: {list(self.model.models.keys())}"
            )

        print(f"[WakeWord] Loaded: {model_path}")
        print(f"[WakeWord] Threshold: {self.threshold}")

    def wait_for_wakeword(self, microphone):

        print("[WakeWord] Listening for ASTA...")

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

            score = float(prediction["asta"])

            if self.debug:
                print(
                    f"\r[WakeWord] ASTA: {score:.3f}",
                    end="",
                    flush=True,
                )

            if score >= self.threshold:
                print(
                    f"\n[WakeWord] ASTA detected "
                    f"(score={score:.3f})"
                )

                return microphone.get_buffer()