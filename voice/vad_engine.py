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
        min_speech_duration=0.45,
        threshold=0.65,
        silence_ms=800,
        speech_pad_ms=300,
        min_rms=0.025,
        min_peak=0.08,
        start_chunk_rms=0.008,
    ):

        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_rms = min_rms
        self.min_peak = min_peak
        self.start_chunk_rms = start_chunk_rms

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
        recording = False

        # Do not feed the wake-word tail into Whisper. The wake-word detector's
        # ring buffer is useful for its own detection, but its final samples
        # can contain the wake phrase rather than the user's command.
        # Command audio starts fresh after the wake-word confirmation.
        _ = initial_audio

        try:
            while True:
                try:
                    chunk = microphone.get_chunk().flatten()
                except queue.Empty:
                    continue

                if not recording and time.monotonic() - start_wait > speech_timeout:
                    print("[VAD] No command after wakeword.")
                    return None

                chunk = np.asarray(chunk, dtype=np.float32)
                chunk_rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

                # Prevent very-low-level room/device noise from triggering the
                # speech state. Once speech has started, keep the real audio.
                vad_chunk = (
                    chunk
                    if chunk_rms >= self.start_chunk_rms
                    else np.zeros_like(chunk)
                )

                tensor = torch.from_numpy(vad_chunk).float()
                event = self.vad(tensor)

                if self.debug:
                    print(event)

                if not recording and self.is_speech_started(event):
                    print("[VAD] Command started.")
                    recording = True

                if recording:
                    audio_buffer.append(chunk)

                if recording and self.is_speech_ended(event):
                    print("[VAD] Command finished.")
                    break

        finally:
            self.vad.reset_states()

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        duration = len(audio) / self.sample_rate

        if self.debug:
            print(
                f"[VAD] RMS={rms:.4f} peak={peak:.4f} "
                f"duration={duration:.3f}s"
            )

        if rms < self.min_rms or peak < self.min_peak:
            print(
                f"[VAD] Low-energy command "
                f"(rms={rms:.4f}, peak={peak:.4f})."
            )
            return None

        minimum_samples = int(self.sample_rate * self.min_speech_duration)
        if len(audio) < minimum_samples:
            print(f"[VAD] Command too short ({duration:.2f}s).")
            return None

        return np.ascontiguousarray(audio, dtype=np.float32)
