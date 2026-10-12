"""Tell the user's barge-in apart from ASTA's own voice in the microphone.

While ASTA speaks, the microphone hears the speakers. The old detector took a
single 160 ms sample as the "echo level" and interrupted on two frames (80 ms)
louder than 1.7x that. A quiet first syllable followed by a normal word was
enough to cut ASTA off mid-sentence. This gate instead:

* learns the loudest echo seen during a longer warm-up and keeps tracking it
  (a slowly decaying peak envelope, so loud words raise it),
* absorbs short bursts that never become a sustained onset into the envelope,
* needs a longer sustained rise during speech (default 240 ms, one 40 ms
  dip allowed) before it reports an interruption.

Tunables: ASTA_BARGE_CONFIRM_MS (240), ASTA_BARGE_WARMUP_MS (450).
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np


def _env_ms(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default)))) / 1000.0
    except (TypeError, ValueError):
        return default / 1000.0


class EchoGate:
    def __init__(
        self,
        *,
        min_rms: float = 0.012,
        min_peak: float = 0.045,
        idle_rms_ratio: float = 1.35,
        idle_peak_ratio: float = 1.20,
        tts_rms_ratio: float = 1.70,
        tts_peak_ratio: float = 1.45,
        idle_confirm_frames: int = 2,
        tts_confirm_frames: int = 6,
        decay: float = 0.985,
    ) -> None:
        self.min_rms = min_rms
        self.min_peak = min_peak
        self.idle_rms_ratio = idle_rms_ratio
        self.idle_peak_ratio = idle_peak_ratio
        self.tts_rms_ratio = tts_rms_ratio
        self.tts_peak_ratio = tts_peak_ratio
        self.idle_confirm_frames = max(1, int(idle_confirm_frames))
        self.tts_confirm_frames = max(1, int(tts_confirm_frames))
        self.decay = decay
        self.reset()

    def reset(self) -> None:
        self.echo_rms: float | None = None
        self.echo_peak: float | None = None
        self._hits: deque[bool] = deque(maxlen=self.tts_confirm_frames + 1)
        self._burst: list[tuple[float, float]] = []

    def _learn(self, rms: float, peak: float) -> None:
        self.echo_rms = max(rms, (self.echo_rms or 0.0) * self.decay)
        self.echo_peak = max(peak, (self.echo_peak or 0.0) * self.decay)

    def update(self, rms: float, peak: float, *, tts_active: bool, warming: bool = False) -> bool:
        """Feed one analysis frame; True when the user is really talking."""
        if tts_active:
            if warming or self.echo_rms is None:
                self._learn(rms, peak)
                self._hits.clear()
                return False
            gate_rms = max(self.min_rms, self.echo_rms * self.tts_rms_ratio)
            gate_peak = max(self.min_peak, self.echo_peak * self.tts_peak_ratio)
            needed, tolerance = self.tts_confirm_frames, 1
        else:
            gate_rms = max(self.min_rms, (self.echo_rms or 0.0) * self.idle_rms_ratio)
            gate_peak = max(self.min_peak, (self.echo_peak or 0.0) * self.idle_peak_ratio)
            needed, tolerance = self.idle_confirm_frames, 0

        onset = rms >= gate_rms and peak >= gate_peak
        self._hits.append(onset)
        recent = list(self._hits)[-(needed + tolerance):]
        if onset:
            self._burst.append((rms, peak))
            if len(recent) >= needed and sum(recent) >= needed:
                return True
            return False

        if tts_active:
            if self._burst and not any(recent[-(tolerance + 1):]):
                # A loud moment that faded before it became an onset was
                # ASTA's own emphasis; remember it as echo.
                self._learn(max(item[0] for item in self._burst) * 0.9,
                            max(item[1] for item in self._burst) * 0.9)
                self._burst.clear()
            self._learn(rms, peak)
        elif not any(recent):
            self._burst.clear()
        return False


def voiced_fraction(audio, sample_rate: int, *, frame_seconds: float = 0.04,
                    min_rms: float = 0.012) -> float | None:
    """Share of short frames that sound like a voice (None: too little audio)."""
    samples = np.asarray(audio, dtype=np.float32).flatten()
    frame = max(1, int(sample_rate * frame_seconds))
    count = samples.size // frame
    if count < 5:
        return None
    frames = samples[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.mean(rms >= min_rms))


def confirm_frames_from_env(interval_seconds: float) -> int:
    return max(2, round(_env_ms("ASTA_BARGE_CONFIRM_MS", 240) / max(interval_seconds, 0.001)))


def warmup_seconds_from_env() -> float:
    return _env_ms("ASTA_BARGE_WARMUP_MS", 450)
