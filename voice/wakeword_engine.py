from collections import deque
from pathlib import Path

import numpy as np
from openwakeword.model import Model


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "models"
)

DEFAULT_MODELS = [
    MODEL_DIR / "hello_asta.onnx",
    MODEL_DIR / "hey_asta.onnx",
    MODEL_DIR / "wake_up_asta.onnx",
]


class WakeWordEngine:

    def __init__(
        self,
        model_paths=None,
        threshold=0.3,
        debug=True,
    ):
        self.model_paths = [
            str(path)
            for path in (model_paths or DEFAULT_MODELS)
        ]

        self.threshold = threshold
        self.debug = debug

        # ---------------------------------------------------------
        # Load all three wake-word models
        # ---------------------------------------------------------

        self.model = Model(
            wakeword_models=self.model_paths,
            inference_framework="onnx",
            vad_threshold=0,
        )

        self.prediction_history = deque(maxlen=5)

        self.wakewords = [
            "hello_asta",
            "hey_asta",
            "wake_up_asta",
        ]

        # Last detected wake word.
        #
        # IMPORTANT:
        # wait_for_wakeword() still returns ONLY the microphone
        # buffer so the existing VAD pipeline remains compatible.
        self.last_detected_word = None

        # ---------------------------------------------------------
        # Verify every expected model was loaded
        # ---------------------------------------------------------

        loaded_models = list(self.model.models.keys())

        missing = [
            word
            for word in self.wakewords
            if word not in loaded_models
        ]

        if missing:
            raise RuntimeError(
                "Wake-word models were not loaded.\n"
                f"Missing: {missing}\n"
                f"Loaded models: {loaded_models}"
            )

        print(
            "[WakeWord] Loaded: "
            + ", ".join(self.wakewords)
        )

        print(
            f"[WakeWord] Threshold: "
            f"{self.threshold}"
        )

    # -------------------------------------------------------------
    # Wait for any wake word
    # -------------------------------------------------------------

    def wait_for_wakeword(self, microphone):

        print(
            "[WakeWord] Listening for: "
            + ", ".join(self.wakewords)
        )

        buffer = deque()

        while True:

            chunk = microphone.get_chunk().flatten()

            buffer.extend(chunk)

            # openWakeWord processes 1280 samples at a time.
            if len(buffer) < 1280:
                continue

            audio = np.array(
                [
                    buffer.popleft()
                    for _ in range(1280)
                ],
                dtype=np.float32,
            )

            # -----------------------------------------------------
            # Microphone provides normalized float audio.
            # Convert to int16 for openWakeWord.
            # -----------------------------------------------------

            audio_int16 = np.clip(
                audio * 32767,
                -32768,
                32767,
            ).astype(np.int16)

            prediction = self.model.predict(
                audio_int16
            )

            # -----------------------------------------------------
            # Read scores for all three wake words.
            # -----------------------------------------------------

            scores = {
                word: float(
                    prediction.get(word, 0.0)
                )
                for word in self.wakewords
            }

            # -----------------------------------------------------
            # Debug output
            # -----------------------------------------------------

            if self.debug:

                print(
                    "\r"
                    + " | ".join(
                        f"{word}: {score:.3f}"
                        for word, score in scores.items()
                    ),
                    end="",
                    flush=True,
                )

            # -----------------------------------------------------
            # Find the strongest wake word.
            # -----------------------------------------------------

            detected_word = max(
                scores,
                key=scores.get,
            )

            detected_score = scores[
                detected_word
            ]

            # -----------------------------------------------------
            # Wake word detected
            # -----------------------------------------------------

            if detected_score >= self.threshold:

                self.last_detected_word = (
                    detected_word
                )

                print(
                    f"\n[WakeWord] "
                    f"{detected_word} detected "
                    f"(score={detected_score:.3f})"
                )

                # IMPORTANT:
                # Keep the original interface.
                #
                # test_voice.py expects:
                #
                #     initial_audio = wake.wait_for_wakeword(mic)
                #
                # and passes that directly to VAD.
                #
                # Therefore DO NOT return:
                #
                #     detected_word, microphone.get_buffer()
                #
                # Return only the audio buffer.
                return microphone.get_buffer()