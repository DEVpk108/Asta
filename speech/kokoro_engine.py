import time

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline


class KokoroEngine:
    """Local Kokoro TTS backend with GPU-first, streaming playback."""

    def __init__(self, voice="am_michael", speed=1.0, lang_code="a", device=None):
        self.voice = voice
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = KPipeline(lang_code=lang_code, device=self.device)

        print(
            f"[Speech] Kokoro ready (voice={self.voice}, device={self.device})",
            flush=True,
        )

    def speak(self, text):
        if not text:
            return

        start = time.perf_counter()
        first_audio_time = None
        total_samples = 0

        try:
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r"(?<=[.!?])\s+",
            )

            with sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype="float32",
            ) as stream:
                for _, _, audio in generator:
                    if audio is None:
                        continue

                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()

                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if audio.size == 0:
                        continue

                    if first_audio_time is None:
                        first_audio_time = time.perf_counter() - start
                        print(
                            f"[Speech] Kokoro TTFA: {first_audio_time:.3f}s",
                            flush=True,
                        )

                    stream.write(audio)
                    total_samples += int(audio.size)

        except Exception as exc:
            print(
                f"[Speech] Kokoro error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        elapsed = time.perf_counter() - start
        duration = total_samples / 24000 if total_samples else 0.0
        print(
            f"[Speech] Kokoro playback: {elapsed:.2f}s for {duration:.2f}s audio",
            flush=True,
        )
