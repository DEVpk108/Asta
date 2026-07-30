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
        
        
        print("[VAD] Waiting for speech...")
        start_wait = time.monotonic()

        audio_buffer = []
        recording = False
        if initial_audio is not None:
            audio_buffer.append(initial_audio)
            recording = True
            print("[VAD] Continuing from wakeword...")

        try:
            while True:
                try:

                    chunk = microphone.get_chunk().flatten()
                except queue.Empty:
                    continue
                
                if not recording:
                    if time.monotonic() - start_wait > speech_timeout:
                        print("[VAD] Speech timeout.")
                        return None

                tensor = torch.from_numpy(chunk).float()

                event = self.vad(tensor)
            
                if self.debug:
                    print(event)

                # Speech started
                if self.is_speech_started(event):

                    print("[VAD] Speech detected.")

                    recording = True
                    if initial_audio is not None:
                        print("[VAD] Using wakeword buffer.")           

                if recording:
                    audio_buffer.append(chunk)

                # Speech ended
                if recording and self.is_speech_ended(event):

                    print("[VAD] End of speech.")

                    break
        finally:
            self.vad.reset_states()

        if not audio_buffer:
            return None
        
        audio = np.concatenate(audio_buffer)
        rms = np.sqrt(np.mean(audio ** 2))
        
        if self.debug:
            peak = np.max(np.abs(audio))
            print(f"[VAD] Peak : {peak:.4f}")

            print(f"[VAD] RMS : {rms:.4f}")
        
        if rms < 0.02:

            print("[VAD] Low energy segment ignored.")

            return None

        

        minimum_samples = int(
            self.sample_rate * self.min_speech_duration
        )

        if len(audio) < minimum_samples:
            return None
        
        duration = len(audio) / self.sample_rate

        minimum_duration = max(
            self.min_speech_duration,
            0.8,
        )

        if duration < minimum_duration:
            print("[VAD] Too short.")
            return None

        return np.ascontiguousarray(audio, dtype=np.float32)