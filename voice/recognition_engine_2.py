import os
import warnings

# TensorFlow / Abseil logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

# Keras warning
warnings.filterwarnings(
    "ignore",
    message=".*tf\\.placeholder is deprecated.*",
)

import numpy as np
import moonshine


class RecognitionEngine:

    def __init__(self, model_name="moonshine/base"):
        self.model_name = model_name
        self.debug = False

        print(
            f"[Recognition] Loading Moonshine: "
            f"{self.model_name}"
        )

        # Load once at startup instead of loading for every command.
        self.model = moonshine.load_model(
            self.model_name
        )

        print("[Recognition] Moonshine ready")

    def transcribe(self, audio):

        if audio is None:
            return ""

        try:
            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

            audio = np.ascontiguousarray(audio)

            # Moonshine expects [batch, samples].
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            result = moonshine.transcribe(
                audio,
                model=self.model,
            )

            if isinstance(result, (list, tuple)):
                text = " ".join(
                    str(item).strip()
                    for item in result
                    if str(item).strip()
                ).strip()
            else:
                text = str(result).strip()

        except Exception as e:
            print(
                f"[Recognition] "
                f"{type(e).__name__}: {e}"
            )
            return ""

        if self.debug:
            print(f"[Recognition] {text}")

        return text