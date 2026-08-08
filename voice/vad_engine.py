import numpy as np
import torch
import time

import queue
from silero_vad import (
    load_silero_vad,
    VADIterator,
)


class VADEngine:

    def __init__(
        self,
        sample_rate=16000,
        min_speech_duration=0.25,
        threshold=0.75,
        silence_ms=700,
        speech_pad_ms=300,
    ):

        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration

        self.model = load_silero_vad()
        self.debug = False

        self.vad = VADIterator(
            self.model,
            sampling_rate=self.sample_rate,
            threshold=threshold,
            min_silence_duration_ms=silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def is_speech_started(self, event):
        return event is not None and "start" in event

    def is_speech_ended(self, event):
        return event is not None and "end" in event

    def collect_utterance(
        self,
        microphone,
        initial_audio=None,
        speech_timeout=3.0,
    ):

        print("[VAD] Waiting for command...")

        start_wait = time.monotonic()

        audio_buffer = []

    # Wakeword already happened.
    # We DO NOT include it in Whisper input.
        recording = False

        try:
            while True:

                try:
                    chunk = microphone.get_chunk().flatten()
                except queue.Empty:
                    continue

            # Timeout waiting for user to start speaking
                if not recording:
                    if time.monotonic() - start_wait > speech_timeout:
                        print("[VAD] No command after wakeword.")
                        return None

                tensor = torch.from_numpy(chunk).float()
                event = self.vad(tensor)

                if self.debug:
                    print(event)

            # -------------------------
            # Speech begins
            # -------------------------
                if not recording and self.is_speech_started(event):

                    print("[VAD] Command started.")

                    recording = True

                # Optional:
                # include ~250 ms before speech start so we don't
                # clip the first word.
                    if initial_audio is not None:

                        preroll = initial_audio[-4000:]      # ~250 ms
                        audio_buffer.append(preroll)

                if recording:
                    audio_buffer.append(chunk)

            # -------------------------
            # Speech ends
            # -------------------------
                if recording and self.is_speech_ended(event):

                    print("[VAD] Command finished.")
                    break

        finally:
            self.vad.reset_states()

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer).astype(np.float32)

        rms = np.sqrt(np.mean(audio ** 2))

        if self.debug:
            print(f"[VAD] RMS: {rms:.4f}")

        if rms < 0.02:
            print("[VAD] Low-energy command.")
            return None

        minimum_samples = int(
            self.sample_rate * self.min_speech_duration
        )

        if len(audio) < minimum_samples:
            print("[VAD] Command too short.")
            return None

        return np.ascontiguousarray(audio, dtype=np.float32)