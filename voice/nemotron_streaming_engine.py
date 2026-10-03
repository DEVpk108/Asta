"""Streaming speech-to-text with NVIDIA Nemotron 3.5 ASR via sherpa-onnx.

Nemotron 3.5 ASR Streaming 0.6B is a cache-aware FastConformer-RNNT model
covering 40 language-locales, including English and Hindi, with automatic
language detection. Unlike Whisper, it consumes audio incrementally: audio is
decoded while the user is still talking, so the transcript is ready almost
immediately after they stop instead of re-decoding the whole utterance.

The engine runs on CPU by default so the RTX GPU stays free for the LLM and
Kokoro. Download the model with ``python scripts/download_speech_models.py``.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT

DEFAULT_MODEL_NAME = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "speech" / DEFAULT_MODEL_NAME
SAMPLE_RATE = 16000
# Trailing silence fed after the user stops so the model's right context is
# filled and the final tokens are flushed (matches sherpa-onnx examples).
TAIL_PADDING_SECONDS = 0.66


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _find_model_file(model_dir: Path, stem: str) -> Path:
    for candidate in (f"{stem}.int8.onnx", f"{stem}.onnx"):
        path = model_dir / candidate
        if path.is_file():
            return path
    matches = sorted(model_dir.glob(f"{stem}*.onnx"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing {stem}*.onnx in {model_dir}")


class NemotronStream:
    """One utterance. Audio is decoded on a worker thread as it arrives."""

    def __init__(self, engine: "NemotronStreamingEngine", language: str):
        self._engine = engine
        self._stream = engine.recognizer.create_stream()
        if language:
            try:
                self._stream.set_option("language", language)
            except Exception:  # Older sherpa-onnx builds without options.
                pass
        self._audio: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._lock = threading.Lock()
        self._partial = ""
        self._final: str | None = None
        self._done = threading.Event()
        self._cancelled = False
        self.samples = 0
        self._worker = threading.Thread(
            target=self._run, name="NemotronStream", daemon=True
        )
        self._worker.start()

    def accept(self, samples) -> None:
        if self._done.is_set() or samples is None:
            return
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size:
            self.samples += int(chunk.size)
            self._audio.put(chunk.copy())

    @property
    def partial_text(self) -> str:
        with self._lock:
            return self._partial

    def finish(self, timeout: float = 10.0) -> str:
        """Flush the stream and return the final transcript."""
        if not self._done.is_set():
            self._audio.put(None)
        self._done.wait(timeout)
        with self._lock:
            return (self._final if self._final is not None else self._partial).strip()

    def cancel(self) -> None:
        self._cancelled = True
        self._audio.put(None)

    def _decode_ready(self) -> None:
        recognizer = self._engine.recognizer
        with self._engine.decode_lock:
            while recognizer.is_ready(self._stream):
                recognizer.decode_stream(self._stream)
            text = recognizer.get_result(self._stream)
        if isinstance(text, str):
            with self._lock:
                self._partial = " ".join(text.split())

    def _run(self) -> None:
        try:
            while True:
                chunk = self._audio.get()
                if chunk is None:
                    break
                # Batch whatever is already queued to keep up with real time.
                parts = [chunk]
                finished = False
                while True:
                    try:
                        extra = self._audio.get_nowait()
                    except queue.Empty:
                        break
                    if extra is None:
                        finished = True
                        break
                    parts.append(extra)
                self._stream.accept_waveform(SAMPLE_RATE, np.concatenate(parts))
                self._decode_ready()
                if finished:
                    break

            if not self._cancelled:
                self._stream.accept_waveform(
                    SAMPLE_RATE,
                    np.zeros(int(SAMPLE_RATE * TAIL_PADDING_SECONDS), dtype=np.float32),
                )
                self._stream.input_finished()
                self._decode_ready()
            with self._lock:
                self._final = self._partial
        except Exception as exc:
            print(
                f"[STT/Nemotron] Stream error: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            self._done.set()


class NemotronStreamingEngine:
    """Owns the sherpa-onnx recognizer; create one stream per utterance."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        language: str | None = None,
        num_threads: int | None = None,
        provider: str | None = None,
    ):
        import sherpa_onnx

        self.model_dir = Path(
            model_dir or os.getenv("ASTA_NEMOTRON_MODEL_DIR") or DEFAULT_MODEL_DIR
        )
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Nemotron model not found at {self.model_dir}. "
                "Run: python scripts/download_speech_models.py"
            )
        self.language = (
            language or os.getenv("ASTA_STT_LANGUAGE", "auto")
        ).strip() or "auto"
        self.num_threads = max(
            1, num_threads or _env_int("ASTA_NEMOTRON_THREADS", 4)
        )
        self.provider = (
            provider or os.getenv("ASTA_NEMOTRON_PROVIDER", "cpu")
        ).strip().lower() or "cpu"
        self.decode_lock = threading.Lock()

        start = time.perf_counter()
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self.model_dir / "tokens.txt"),
            encoder=str(_find_model_file(self.model_dir, "encoder")),
            decoder=str(_find_model_file(self.model_dir, "decoder")),
            joiner=str(_find_model_file(self.model_dir, "joiner")),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            model_type="nemo_transducer",
            provider=self.provider,
        )
        self._warmup()
        print(
            f"[STT/Nemotron] Ready (model={self.model_dir.name}, "
            f"language={self.language}, provider={self.provider}, "
            f"threads={self.num_threads}, "
            f"load={time.perf_counter() - start:.2f}s)",
            flush=True,
        )

    def _warmup(self) -> None:
        stream = self.create_stream()
        stream.accept(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        stream.finish(timeout=30.0)

    def create_stream(self, language: str | None = None) -> NemotronStream:
        return NemotronStream(self, language or self.language)

    def transcribe(self, audio) -> str:
        stream = self.create_stream()
        stream.accept(audio)
        return stream.finish(timeout=60.0)
