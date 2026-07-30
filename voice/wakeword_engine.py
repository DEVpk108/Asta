from openwakeword.model import Model
import numpy as np
from collections import deque

class WakeWordEngine:

    def __init__(self):
        self.model = Model(
            inference_framework="onnx",
            vad_threshold=0,
        )

        self.threshold = 0.09
        self.debug = True

    def wait_for_wakeword(self, microphone):

        print("[WakeWord] Listening...")

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
            audio_int16 = (audio * 32767).astype(np.int16)

            prediction = self.model.predict(audio_int16)
            if self.debug:
                score = prediction["hey_jarvis"]

                print(f"\rScore: {score:.3f}", end="")

            if score >= self.threshold:
               print("\nWakeword detected!")
               return microphone.get_buffer()
        