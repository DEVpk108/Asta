"""Smart Turn v3 end-of-turn detection (pipecat-ai/smart-turn, BSD-2).

Silero VAD can only tell that the user is silent. Smart Turn looks at the
last 8 seconds of the utterance and predicts whether the speaker actually
finished (23 languages, including English and Hindi). With it, A.S.T.A. can
end a turn after ~200 ms of silence when the sentence sounds complete, and
still waits the normal ``ASTA_VAD_SILENCE_MS`` when it sounds unfinished.

The Whisper log-mel features are computed with NumPy (verified to match
``transformers.WhisperFeatureExtractor``), so ``transformers`` is not needed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "speech" / "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
N_FFT = 400
HOP = 160
N_MELS = 80


def _hz_to_mel(freq):
    freq = np.asarray(freq, dtype=np.float64)
    linear = 3.0 * freq / 200.0
    log_part = 15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0) * (27.0 / np.log(6.4))
    return np.where(freq >= 1000.0, log_part, linear)


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    linear = 200.0 * mel / 3.0
    log_part = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel - 15.0))
    return np.where(mel >= 15.0, log_part, linear)


def _mel_filters() -> np.ndarray:
    """Slaney-normalized mel filter bank, identical to Whisper's."""
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    hz = _mel_to_hz(mel_points)
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz[2 : N_MELS + 2] - hz[:N_MELS])
    return weights * enorm[:, None]


_MEL = _mel_filters()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def whisper_features(audio) -> np.ndarray:
    """Whisper log-mel features for exactly 8 s of 16 kHz audio (80 x 800)."""
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    target = WINDOW_SECONDS * SAMPLE_RATE
    if samples.size > target:
        samples = samples[-target:]
    elif samples.size < target:
        samples = np.pad(samples, (target - samples.size, 0))
    samples = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (padded.size - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)) ** 2
    log_mel = np.log10(np.maximum(_MEL @ power.T, 1e-10))[:, :-1]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0).astype(np.float32)


class SmartTurnDetector:
    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None):
        import onnxruntime as ort

        self.model_path = Path(
            model_path or os.getenv("ASTA_SMART_TURN_MODEL") or DEFAULT_MODEL_PATH
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Smart Turn model not found at {self.model_path}. "
                "Run: python scripts/download_speech_models.py"
            )
        if threshold is None:
            try:
                threshold = float(os.getenv("ASTA_SMART_TURN_THRESHOLD", "0.5"))
            except ValueError:
                threshold = 0.5
        self.threshold = min(0.99, max(0.01, float(threshold)))

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.completion_probability(np.zeros(SAMPLE_RATE, dtype=np.float32))
        print(f"[SmartTurn] Ready ({self.model_path.name}, threshold={self.threshold:.2f})", flush=True)

    def completion_probability(self, audio) -> float:
        features = whisper_features(audio)[None, :, :]
        output = self.session.run(None, {"input_features": features})[0]
        return float(np.asarray(output).reshape(-1)[0])

    def is_complete(self, audio) -> bool:
        start = time.perf_counter()
        probability = self.completion_probability(audio)
        complete = probability >= self.threshold
        print(
            f"[SmartTurn] {'complete' if complete else 'incomplete'} "
            f"p={probability:.2f} ({(time.perf_counter() - start) * 1000:.0f} ms)",
            flush=True,
        )
        return complete


def load_smart_turn() -> SmartTurnDetector | None:
    """Return a detector when enabled and installed, otherwise ``None``."""
    if str(os.getenv("ASTA_SMART_TURN", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return None
    try:
        return SmartTurnDetector()
    except Exception as exc:
        print(f"[SmartTurn] Disabled: {type(exc).__name__}: {exc}", flush=True)
        return None


class TurnEndTracker:
    """Decides when an utterance is over, given Silero VAD end events.

    Without Smart Turn the first VAD end event finishes the turn (the old
    behaviour). With Smart Turn, Silero fires early (short silence); if the
    speech sounds complete the turn ends immediately, otherwise the tracker
    keeps listening until the full ``ASTA_VAD_SILENCE_MS`` has elapsed or the
    user resumes speaking.
    """

    def __init__(
        self,
        detector: SmartTurnDetector | None,
        extra_wait_samples: int,
        *,
        hold=None,
        short_speech_samples: int = 0,
        short_threshold: float | None = None,
    ):
        self.detector = detector
        self.extra_wait_samples = max(0, int(extra_wait_samples))
        self.pending = False
        self.waited = 0
        # Optional callable: True while the live transcript ends on a verb or
        # connector ("open", "search for", "and"), i.e. clearly unfinished.
        self.hold = hold
        # Very short turns ("open" + pause) need a more confident Smart Turn.
        self.short_speech_samples = max(0, int(short_speech_samples))
        self.short_threshold = short_threshold
        self.speech_samples = 0

    def _sounds_complete(self, audio) -> bool:
        detector = self.detector
        threshold = getattr(detector, "threshold", 0.5)
        short = (
            self.short_threshold is not None
            and self.speech_samples < self.short_speech_samples
        )
        probability_fn = getattr(detector, "completion_probability", None)
        if not callable(probability_fn) or not isinstance(threshold, (int, float)):
            return detector.is_complete(audio)
        start = time.perf_counter()
        probability = probability_fn(audio)
        required = max(threshold, self.short_threshold) if short else threshold
        complete = probability >= required
        print(
            f"[SmartTurn] {'complete' if complete else 'incomplete'} "
            f"p={probability:.2f}{' (short turn, need %.2f)' % required if short else ''} "
            f"({(time.perf_counter() - start) * 1000:.0f} ms)",
            flush=True,
        )
        return complete

    def update(self, *, started: bool, ended: bool, chunk_samples: int, get_audio) -> bool:
        self.speech_samples += int(chunk_samples)
        if self.pending:
            if started:
                self.pending = False
                self.waited = 0
                return False
            self.waited += int(chunk_samples)
            return self.waited >= self.extra_wait_samples
        if not ended:
            return False
        if self.detector is None or self.extra_wait_samples == 0:
            return True
        if self.hold is not None:
            try:
                unfinished = bool(self.hold())
            except Exception:
                unfinished = False
            if unfinished:
                print("[SmartTurn] holding: phrase sounds unfinished", flush=True)
                self.pending = True
                self.waited = 0
                return False
        try:
            if self._sounds_complete(get_audio()):
                return True
        except Exception as exc:
            print(f"[SmartTurn] Prediction failed: {type(exc).__name__}: {exc}", flush=True)
            return True
        self.pending = True
        self.waited = 0
        return False

