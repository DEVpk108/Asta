import os
import queue
import time
from collections import deque

import numpy as np
import torch
from .smart_turn import TurnEndTracker, load_smart_turn
from silero_vad import (
    VADIterator,
    load_silero_vad,
)


class VADEngine:

    def __init__(
        self,
        sample_rate=16000,
        min_speech_duration=0.30,
        threshold=0.50,
        silence_ms=None,
        speech_pad_ms=500,
        min_rms=0.012,
        min_peak=0.04,
        start_chunk_rms=0.005,
        pre_roll_ms=None,
    ):

        if silence_ms is None:
            try:
                silence_ms = int(os.getenv("ASTA_VAD_SILENCE_MS", "700"))
            except ValueError:
                silence_ms = 700
        silence_ms = max(250, min(2000, int(silence_ms)))

        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_rms = min_rms
        self.min_peak = min_peak
        self.start_chunk_rms = start_chunk_rms
        if pre_roll_ms is None:
            try:
                pre_roll_ms = int(os.getenv("ASTA_VAD_PRE_ROLL_MS", "900"))
            except ValueError:
                pre_roll_ms = 800
        pre_roll_ms = max(250, min(1600, int(pre_roll_ms)))
        self.pre_roll_samples = max(1, int(sample_rate * pre_roll_ms / 1000))

        self.model = load_silero_vad()
        self.debug = False

        # Smart Turn lets Silero report silence early; the turn then ends at
        # once when the speech sounds complete and otherwise waits the full
        # silence_ms. Without the model, behaviour is unchanged.
        self.silence_ms = silence_ms
        self.smart_turn = load_smart_turn()
        iterator_silence_ms = silence_ms
        if self.smart_turn is not None:
            try:
                short_ms = int(os.getenv("ASTA_SMART_TURN_SILENCE_MS", "200"))
            except ValueError:
                short_ms = 200
            iterator_silence_ms = max(100, min(silence_ms, short_ms))
        self.iterator_silence_ms = iterator_silence_ms

        self.vad = VADIterator(
            self.model,
            sampling_rate=self.sample_rate,
            threshold=threshold,
            min_silence_duration_ms=iterator_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def new_turn_tracker(self, hold=None, short_turns=True):
        silence_ms = int(getattr(self, "silence_ms", 700))
        iterator_ms = int(getattr(self, "iterator_silence_ms", silence_ms))
        extra_ms = max(0, silence_ms - iterator_ms)
        try:
            short_ms = int(os.getenv("ASTA_SMART_TURN_SHORT_MS", "900"))
            short_threshold = float(os.getenv("ASTA_SMART_TURN_SHORT_THRESHOLD", "0.85"))
        except ValueError:
            short_ms, short_threshold = 900, 0.85
        return TurnEndTracker(
            getattr(self, "smart_turn", None),
            int(self.sample_rate * extra_ms / 1000),
            hold=hold,
            short_speech_samples=int(self.sample_rate * short_ms / 1000) if short_turns else 0,
            short_threshold=short_threshold if short_turns else None,
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
        on_audio=None,
        hold=None,
        short_turns=True,
    ):
        """Record one utterance.

        ``on_audio`` receives every captured chunk as it is recorded so a
        streaming recognizer can decode while the user is still speaking.
        """
        print("[VAD] Waiting for command...")
        try:
            tracker = self.new_turn_tracker(hold=hold, short_turns=short_turns)
        except TypeError:  # test doubles with the old signature
            tracker = self.new_turn_tracker()

        def capture(samples):
            audio_buffer.append(samples)
            if on_audio is not None:
                on_audio(samples)

        start_wait = time.monotonic()
        audio_buffer = []
        pre_roll = deque(maxlen=self.pre_roll_samples)
        recording = False
        recording_started_at = None

        initial_seed = None
        used_initial_seed = False
        if initial_audio is not None:
            seed = np.asarray(initial_audio, dtype=np.float32).flatten()
            if seed.size:
                initial_seed = seed[-self.pre_roll_samples :].copy()

        # Post-TTS speech can begin before the live VAD iterator receives its
        # first chunk. When the handoff seed already contains speech energy,
        # start the utterance from that seed instead of waiting for a fresh
        # Silero start event and losing the first word.
        if initial_seed is not None:
            seed_rms = float(np.sqrt(np.mean(np.square(initial_seed))))
            seed_peak = (
                float(np.max(np.abs(initial_seed)))
                if initial_seed.size
                else 0.0
            )
            seed_gate_rms = max(
                self.start_chunk_rms * 1.5,
                self.min_rms * 0.55,
            )
            seed_gate_peak = max(
                self.min_peak * 0.70,
                0.028,
            )
            if seed_rms >= seed_gate_rms and seed_peak >= seed_gate_peak:
                recording = True
                recording_started_at = time.monotonic()
                capture(initial_seed)
                initial_seed = None
                used_initial_seed = True
                print(
                    "[VAD] Seed contains speech; preserving the full post-TTS onset.",
                    flush=True,
                )

        # The wake-word detector's ring buffer is intentionally not reused for
        # command recognition. The command gets a fresh, live pre-roll instead.
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
                chunk_rms = (
                    float(np.sqrt(np.mean(np.square(chunk))))
                    if chunk.size
                    else 0.0
                )

                # Lower the VAD input gate slightly so soft first words such as
                # "turn" are detected earlier, while the post-capture RMS/peak
                # checks below still reject genuinely low-energy audio.
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
                    recording_started_at = time.monotonic()

                    if initial_seed is not None:
                        capture(initial_seed)
                        initial_seed = None
                    elif pre_roll:
                        capture(np.asarray(pre_roll, dtype=np.float32))
                    started_now = True
                else:
                    started_now = False

                if recording:
                    capture(chunk)
                else:
                    pre_roll.extend(chunk)

                if recording and tracker.update(
                    started=started_now or self.is_speech_started(event),
                    ended=self.is_speech_ended(event),
                    chunk_samples=len(chunk),
                    get_audio=lambda: np.concatenate(audio_buffer),
                ):
                    elapsed = (
                        time.monotonic() - recording_started_at
                        if recording_started_at is not None
                        else 0.0
                    )
                    minimum_recording_seconds = max(
                        self.min_speech_duration,
                        0.35,
                    )
                    if elapsed >= minimum_recording_seconds:
                        print(
                            f"[VAD] Command finished "
                            f"(capture={elapsed:.2f}s)."
                        )
                        break

        finally:
            self.vad.reset_states()

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        duration = len(audio) / self.sample_rate

        print(
            f"[VAD] Capture: duration={duration:.3f}s "
            f"rms={rms:.4f} peak={peak:.4f} "
            f"seed={'yes' if used_initial_seed else 'no'}",
            flush=True,
        )
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
