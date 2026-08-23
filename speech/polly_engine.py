import time

import boto3
import keyboard
import numpy as np
import sounddevice as sd


class PollyEngine:

    def __init__(self):
        self.polly_client = boto3.client(
            "polly"
        )

    def speak(self, text):

        if not text:
            return

        # ---------------------------------------------------------
        # TTS synthesis
        # ---------------------------------------------------------

        synthesis_start = time.perf_counter()

        try:
            response = self.polly_client.synthesize_speech(
                Engine="standard",
                OutputFormat="pcm",
                Text=text,
                VoiceId="Matthew",
            )

            audio_bytes = (
                response["AudioStream"].read()
            )

            synthesis_time = (
                time.perf_counter()
                - synthesis_start
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - synthesis_start
            )

            print(
                f"[Speech] Synthesis error "
                f"after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            return

        # ---------------------------------------------------------
        # Audio preparation
        # ---------------------------------------------------------

        audio_array = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        )

        # ---------------------------------------------------------
        # Playback
        # ---------------------------------------------------------

        playback_start = time.perf_counter()

        try:
            sd.play(
                audio_array,
                samplerate=13500,
            )

            keyboard.add_hotkey(
                "space",
                sd.stop,
            )

            sd.wait()

        finally:
            playback_time = (
                time.perf_counter()
                - playback_start
            )

        print(
            f"[Speech] Synthesis: "
            f"{synthesis_time:.2f}s",
            flush=True,
        )

        print(
            f"[Speech] Playback: "
            f"{playback_time:.2f}s",
            flush=True,
        )