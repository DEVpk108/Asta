import time
import numpy as np


class DatasetRecorder:

    def __init__(self, microphone, vad):

        self.microphone = microphone
        self.vad = vad

    # --------------------------------------------------

    def countdown(self):

        print()

        for i in (3, 2, 1):

            print(i)

            time.sleep(1)

    # --------------------------------------------------

    def beep(self):

        # Console beep
        print("\a", end="", flush=True)

    # --------------------------------------------------

    def record(self, prompt):

        print("-" * 60)
        print(f'Say: "{prompt}"')
        print("-" * 60)

        self.countdown()

        print("\n🎤 Speak now...\n")

        self.beep()

        start = time.time()

        audio = self.vad.collect_utterance(
            self.microphone,
            initial_audio=None,
            speech_timeout=5.0,
        )

        if audio is None:

            print("[Recorder] No speech detected.")

            return None

        duration = len(audio) / self.microphone.sample_rate

        elapsed = time.time() - start

        print()

        print(f"[Recorder] Recording : {duration:.2f} sec")
        print(f"[Recorder] Total Time : {elapsed:.2f} sec")

        return np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )