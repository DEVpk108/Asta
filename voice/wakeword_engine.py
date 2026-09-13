from collections import deque
from pathlib import Path

import numpy as np
from openwakeword.model import Model


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "ai" / "wakeword" / "generated" / "models"

DEFAULT_MODELS = [
    MODEL_DIR / "hello_asta.onnx",
    MODEL_DIR / "hey_asta.onnx",
    MODEL_DIR / "wake_up_asta.onnx",
]


class WakeWordEngine:

    def __init__(
        self,
        model_paths=None,
        threshold=0.35,
        debug=True,
        confirmation_frames=2,
        strong_threshold=0.85,
        silence_rms=0.0003,
    ):
        self.model_paths = [str(path) for path in (model_paths or DEFAULT_MODELS)]
        self.threshold = float(threshold)
        self.debug = debug
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.strong_threshold = max(self.threshold, float(strong_threshold))
        self.silence_rms = float(silence_rms)

        self.model = Model(
            wakeword_models=self.model_paths,
            inference_framework="onnx",
            vad_threshold=0,
        )

        self.wakewords = ["hello_asta", "hey_asta", "wake_up_asta"]
        self.last_detected_word = None

        loaded_models = list(self.model.models.keys())
        missing = [word for word in self.wakewords if word not in loaded_models]
        if missing:
            raise RuntimeError(
                "Wake-word models were not loaded.\n"
                f"Missing: {missing}\n"
                f"Loaded models: {loaded_models}"
            )

        print("[WakeWord] Loaded: " + ", ".join(self.wakewords))
        print(f"[WakeWord] Threshold: {self.threshold:.2f}")
        print(
            f"[WakeWord] Confirmation: {self.confirmation_frames} consecutive "
            f"frame(s); strong score >= {self.strong_threshold:.2f} still "
            "requires confirmation"
        )
        print(f"[WakeWord] Silence gate: rms < {self.silence_rms:.4f}")

    def wait_for_wakeword(self, microphone, should_continue=None):
        """Wait for a confirmed wake word until the caller asks the listener to pause."""
        print("[WakeWord] Listening for: " + ", ".join(self.wakewords))

        buffer = deque()
        score_history = deque(maxlen=self.confirmation_frames)
        if should_continue is None:
            should_continue = lambda: True

        candidate_word = None
        candidate_hits = 0
        diagnostic_windows = 0

        while True:
            if not should_continue():
                return None

            chunk = microphone.get_chunk().flatten()
            buffer.extend(chunk)

            if len(buffer) < 1280:
                continue

            audio = np.array(
                [buffer.popleft() for _ in range(1280)],
                dtype=np.float32,
            )

            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            diagnostic_windows += 1

            if self.debug and diagnostic_windows % 20 == 0:
                print(
                    f"\n[WakeWord] Audio RMS={rms:.5f}, peak={peak:.5f}",
                    flush=True,
                )

            # Reject only the very-low-energy background/silence region. The
            # measured false activation was around RMS=0.00018, while a real
            # wakeword window measured around RMS=0.00088, so 0.00030 is a
            # conservative separation without recreating the old over-gate.
            if rms < self.silence_rms:
                candidate_word = None
                candidate_hits = 0
                score_history.clear()
                continue

            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            prediction = self.model.predict(audio_int16)

            scores = {
                word: float(prediction.get(word, 0.0))
                for word in self.wakewords
            }

            if self.debug:
                print(
                    "\r" + " | ".join(
                        f"{word}: {score:.3f}" for word, score in scores.items()
                    ),
                    end="",
                    flush=True,
                )

            detected_word = max(scores, key=scores.get)
            detected_score = scores[detected_word]

            if detected_score < self.threshold:
                candidate_word = None
                candidate_hits = 0
                score_history.clear()
                continue

            if detected_word == candidate_word:
                candidate_hits += 1
            else:
                candidate_word = detected_word
                candidate_hits = 1
                score_history.clear()

            score_history.append(detected_score)

            if candidate_hits < self.confirmation_frames:
                continue

            mean_score = float(np.mean(score_history)) if score_history else 0.0
            if mean_score < self.threshold:
                candidate_word = None
                candidate_hits = 0
                score_history.clear()
                continue

            if not should_continue():
                return None

            self.last_detected_word = detected_word
            print(
                f"\n[WakeWord] {detected_word} detected "
                f"(score={detected_score:.3f}, mean={mean_score:.3f}, "
                f"confirmed_frames={candidate_hits})"
            )

            return microphone.get_buffer()
