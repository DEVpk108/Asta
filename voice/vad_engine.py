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
        threshold=0.80,
        silence_ms=700,
        speech_pad_ms=250,
        min_rms=0.025,
        min_peak=0.08,
    ):

        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_rms = min_rms
        self.min_peak = min_peak

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

        # Wakeword already happened. Do not include it in Whisper input.
        try:
            while True:
                try:
                    chunk = microphone.get_chunk().flatten()
                except queue.Empty:
                    continue

                if not recording and time.monotonic() - start_wait > speech_timeout:
                    print("[VAD] No command after wakeword.")
                    return None

                tensor = torch.from_numpy(chunk).float()
                event = self.vad(tensor)

                if self.debug:
                    print(event)

                if not recording and self.is_speech_started(event):
                    print("[VAD] Command started.")
                    recording = True

                    if initial_audio is not None:
                        preroll = initial_audio[-4000:]
                        if len(preroll):
                            audio_buffer.append(preroll)

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

        # Do not send transient/noise captures to Whisper.
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
