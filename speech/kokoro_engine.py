import time

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline


class KokoroEngine:
    """Local Kokoro TTS backend with GPU-first synthesis and decoupled playback."""

    SAMPLE_RATE = 24000

    def __init__(self, voice="am_michael", speed=1.0, lang_code="a", device=None):
        self.voice = voice
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = KPipeline(
            lang_code=lang_code,
            device=self.device,
            repo_id="hexgrad/Kokoro-82M",
        )

        voice_start = time.perf_counter()
        self.pipeline.load_voice(self.voice)
        voice_load_time = time.perf_counter() - voice_start

        warmup_start = time.perf_counter()
        warmup_audio = 0
        try:
            for _, _, audio in self.pipeline(
                "Ready.",
                voice=self.voice,
                speed=self.speed,
                split_pattern=r"(?<=[.!?])\s+",
            ):
                if audio is not None:
                    warmup_audio += int(
                        np.asarray(audio.detach().cpu() if hasattr(audio, "detach") else audio).size
                    )
        except Exception as exc:
            print(
                f"[Speech] Kokoro warmup warning: {type(exc).__name__}: {exc}",
                flush=True,
            )

        warmup_time = time.perf_counter() - warmup_start
        print(
            f"[Speech] Kokoro ready (voice={self.voice}, device={self.device}, "
            f"voice_load={voice_load_time:.3f}s, warmup={warmup_time:.3f}s, "
            f"warmup_audio={warmup_audio} samples)",
            flush=True,
        )

    def synthesize(self, text):
        """Generate sentence audio without blocking the playback device."""
        if not text:
            return None

        start = time.perf_counter()
        first_audio_time = None
        chunks = []
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

                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    continue

                if first_audio_time is None:
                    first_audio_time = time.perf_counter() - start
                    print(
                        f"[Speech] Kokoro TTFA: {first_audio_time:.3f}s",
                        flush=True,
                    )

                chunks.append(audio.copy())
                total_samples += int(audio.size)

        except Exception as exc:
            print(
                f"[Speech] Kokoro synthesis error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

        audio = np.concatenate(chunks) if chunks else None
        elapsed = time.perf_counter() - start
        duration = total_samples / self.SAMPLE_RATE if total_samples else 0.0
        print(
            f"[Speech] Kokoro synth: {elapsed:.2f}s | audio: {duration:.2f}s | "
            f"first_audio: {(first_audio_time or 0.0):.3f}s",
            flush=True,
        )
        return audio

    def play(self, audio):
        """Play already-generated audio and keep playback separate from synthesis."""
        if audio is None:
            return

        start = time.perf_counter()
        try:
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size == 0:
                return

            with sd.OutputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            ) as stream:
                stream.write(audio)

            duration = audio.size / self.SAMPLE_RATE
            print(
                f"[Speech] Kokoro playback: {time.perf_counter() - start:.2f}s | "
                f"audio={duration:.2f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Speech] Kokoro playback error: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def speak(self, text):
        """Compatibility helper for callers that still want synthesize-then-play."""
        audio = self.synthesize(text)
        self.play(audio)
