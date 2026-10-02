from __future__ import annotations

import os
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class IncrementalSessionResult:
    audio: np.ndarray | None
    transcript: str
    remainder: str
    commits: tuple
    duration_seconds: float


class IncrementalSpeechSession:
    """Capture one utterance while repeatedly decoding stable partial speech."""

    def __init__(
        self,
        *,
        vad_engine,
        recognition_engine,
        command_detector,
        sample_rate: int = 16000,
        partial_interval_seconds: float | None = None,
        min_partial_seconds: float | None = None,
        stt_window_seconds: float | None = None,
        pre_roll_seconds: float = 0.75,
    ):
        self.vad_engine = vad_engine
        self.recognition = recognition_engine
        self.detector = command_detector
        self.sample_rate = int(sample_rate)

        self.partial_interval_seconds = self._env_float(
            "ASTA_INCREMENTAL_STT_INTERVAL_MS",
            800.0,
        ) / 1000.0
        self.min_partial_seconds = self._env_float(
            "ASTA_INCREMENTAL_MIN_AUDIO_MS",
            850.0,
        ) / 1000.0
        self.stt_window_seconds = self._env_float(
            "ASTA_INCREMENTAL_STT_WINDOW_MS",
            5000.0,
        ) / 1000.0

        if partial_interval_seconds is not None:
            self.partial_interval_seconds = max(
                0.20,
                float(partial_interval_seconds),
            )
        if min_partial_seconds is not None:
            self.min_partial_seconds = max(
                0.30,
                float(min_partial_seconds),
            )
        if stt_window_seconds is not None:
            self.stt_window_seconds = max(
                1.0,
                float(stt_window_seconds),
            )

        self.pre_roll_seconds = max(0.20, float(pre_roll_seconds))

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    def _should_continue(self, callback):
        return callback is None or bool(callback())

    def _seed_contains_speech(self, seed: np.ndarray) -> bool:
        if seed.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(seed))))
        peak = float(np.max(np.abs(seed)))
        return (
            rms >= max(
                float(self.vad_engine.start_chunk_rms) * 1.5,
                float(self.vad_engine.min_rms) * 0.55,
            )
            and peak >= max(float(self.vad_engine.min_peak) * 0.70, 0.028)
        )

    def run(
        self,
        microphone,
        *,
        initial_audio=None,
        should_continue: Callable[[], bool] | None = None,
        speech_timeout: float = 3.0,
        on_commit: Callable[[object], None] | None = None,
    ) -> IncrementalSessionResult | None:
        start_wait = time.monotonic()
        recording = False
        recording_started_at = None
        # Partial STT cadence is measured in captured audio, not wall-clock
        # time. This keeps scheduling deterministic when STT or the microphone
        # callback runs at a different pace than real time.
        next_partial_audio_seconds = 0.0
        last_partial_audio_seconds = 0.0
        audio_parts: list[np.ndarray] = []
        pre_roll = deque(
            maxlen=max(1, int(self.sample_rate * self.pre_roll_seconds))
        )

        initial_seed = None
        if initial_audio is not None:
            seed = np.asarray(initial_audio, dtype=np.float32).flatten()
            if seed.size:
                seed_samples = min(
                    max(1, int(self.sample_rate * self.pre_roll_seconds)),
                    max(
                        1,
                        int(
                            getattr(
                                self.vad_engine,
                                "pre_roll_samples",
                                int(self.sample_rate * self.pre_roll_seconds),
                            )
                        ),
                    ),
                )
                initial_seed = seed[-seed_samples:].copy()

        reset_states = getattr(self.vad_engine.vad, "reset_states", None)
        if callable(reset_states):
            reset_states()

        try:
            if initial_seed is not None and self._seed_contains_speech(initial_seed):
                recording = True
                recording_started_at = time.monotonic()
                audio_parts.append(initial_seed)
                next_partial_audio_seconds = (
                    self.min_partial_seconds + self.partial_interval_seconds
                )

            while self._should_continue(should_continue):
                try:
                    chunk = np.asarray(
                        microphone.get_chunk().flatten(),
                        dtype=np.float32,
                    )
                except queue.Empty:
                    continue

                now = time.monotonic()
                if (
                    not recording
                    and now - start_wait > float(speech_timeout)
                ):
                    print("[IncrementalVAD] No command after wakeword.", flush=True)
                    return None

                chunk_rms = (
                    float(np.sqrt(np.mean(np.square(chunk))))
                    if chunk.size
                    else 0.0
                )
                vad_chunk = (
                    chunk
                    if chunk_rms >= float(self.vad_engine.start_chunk_rms)
                    else np.zeros_like(chunk)
                )
                event = self.vad_engine.vad(
                    torch.from_numpy(vad_chunk).float()
                )

                if not recording and self.vad_engine.is_speech_started(event):
                    recording = True
                    recording_started_at = now
                    if pre_roll:
                        audio_parts.append(
                            np.asarray(pre_roll, dtype=np.float32)
                        )
                    # Schedule the first partial from captured audio
                    # duration. The current chunk is added immediately below.
                    next_partial_audio_seconds = self.min_partial_seconds

                if recording:
                    audio_parts.append(chunk)
                else:
                    pre_roll.extend(chunk)

                if not recording:
                    continue

                captured_seconds = sum(
                    len(part) for part in audio_parts
                ) / self.sample_rate

                if (
                    captured_seconds >= self.min_partial_seconds
                    and captured_seconds >= next_partial_audio_seconds
                    and captured_seconds > last_partial_audio_seconds
                ):
                    window_samples = max(
                        1,
                        int(self.sample_rate * self.stt_window_seconds),
                    )
                    recent_parts: list[np.ndarray] = []
                    remaining = window_samples

                    for part in reversed(audio_parts):
                        if remaining <= 0:
                            break
                        take = min(len(part), remaining)
                        recent_parts.append(part[-take:])
                        remaining -= take

                    partial_audio = np.concatenate(
                        list(reversed(recent_parts))
                    ).astype(np.float32, copy=False)

                    try:
                        partial_text = self.recognition.transcribe(
                            partial_audio
                        )
                    except Exception as exc:
                        print(
                            "[IncrementalSTT] Partial decode failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        partial_text = ""

                    if partial_text:
                        commit = self.detector.observe(partial_text)
                        if commit is not None:
                            print(
                                "[IncrementalVoice] Committed early action: "
                                f"{commit.text} (source={commit.source_text!r})",
                                flush=True,
                            )
                            if on_commit is not None:
                                on_commit(commit)

                    last_partial_audio_seconds = captured_seconds

                    next_partial_at = now + self.partial_interval_seconds

                if (
                    self.vad_engine.is_speech_ended(event)
                    and captured_seconds >= max(
                        float(self.vad_engine.min_speech_duration),
                        0.30,
                    )
                ):
                    break

            if not audio_parts:
                return None

            audio = np.concatenate(audio_parts).astype(
                np.float32,
                copy=False,
            )
            duration_seconds = len(audio) / self.sample_rate

            if duration_seconds < max(
                float(self.vad_engine.min_speech_duration),
                0.35,
            ):
                return None

            try:
                final_text = self.recognition.transcribe(audio)
            except Exception as exc:
                print(
                    "[IncrementalSTT] Final decode failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                final_text = ""

            remainder = self.detector.finalize(final_text)
            return IncrementalSessionResult(
                audio=audio,
                transcript=final_text,
                remainder=remainder,
                commits=self.detector.commits,
                duration_seconds=duration_seconds,
            )
        finally:
            reset_states = getattr(self.vad_engine.vad, "reset_states", None)
            if callable(reset_states):
                reset_states()
