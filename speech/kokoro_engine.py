import time

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline


class KokoroEngine:
    """Local Kokoro TTS backend with GPU-first inference and direct playback."""

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

        synthesis_start = time.perf_counter()
        first_audio = True
        playback_started = False
        total_samples = 0

        try:
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r"(?<=[.!?])\s+",
            )

            for _, _, audio in generator:
                if audio is None:
                    continue

                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()

                audio = np.asarray(audio, dtype=np.float32)
                if audio.size == 0:
                    continue

                if first_audio:
                    first_audio = False
                    synthesis_time = time.perf_counter() - synthesis_start
                    print(
                        f"[Speech] Kokoro first audio: {synthesis_time:.3f}s",
                        flush=True,
                    )
                    sd.play(audio, samplerate=24000)
                    playback_started = True
                else:
                    sd.play(audio, samplerate=24000, blocking=True)

                total_samples += int(audio.size)

            if playback_started:
                sd.wait()

        except Exception as exc:
            print(
                f"[Speech] Kokoro error: {type(exc).__name__}: {exc}",
                flush=True,
            )

        finally:
            elapsed = time.perf_counter() - synthesis_start
            duration = total_samples / 24000 if total_samples else 0.0
            print(
                f"[Speech] Kokoro total: {elapsed:.2f}s, audio: {duration:.2f}s",
                flush=True,
            )
