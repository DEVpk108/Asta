import os
import queue
import re
import threading
import time
from collections import OrderedDict

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LETTER = re.compile(r"[^\W\d_]")
# Sentence ends for Kokoro's internal splitting (includes the Hindi danda).
_SENTENCE_SPLIT = r"(?<=[.!?\u0964\u0965])\s+"
# Speak the first clause on its own so audio starts sooner on long sentences.
_FIRST_CLAUSE = re.compile(r"[,;:\u2014](?=\s)")


def _env_flag(name, default="1"):
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "no", "off"}


class KokoroEngine:
    """Local Kokoro TTS with streamed, interruptible playback.

    Audio is generated chunk by chunk and written to one persistent output
    stream as soon as each chunk exists, so speech starts after the first
    chunk instead of after the whole sentence. Hindi (Devanagari) text is
    routed to Kokoro's Hindi pipeline automatically.
    """

    SAMPLE_RATE = 24000
    WRITE_SECONDS = 0.02  # ~20 ms writes keep barge-in responsive

    def __init__(self, voice=None, speed=1.0, lang_code="a", device=None):
        self.voice = voice or os.getenv("ASTA_TTS_VOICE", "am_michael")
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = KPipeline(
            lang_code=lang_code,
            device=self.device,
            repo_id="hexgrad/Kokoro-82M",
        )
        self.hindi_enabled = _env_flag("ASTA_TTS_HINDI")
        self.hindi_voice = os.getenv("ASTA_TTS_HINDI_VOICE", "hf_alpha")
        self._hindi_pipeline = None
        self._hindi_failed = False
        self._hindi_lock = threading.Lock()
        self.first_clause_enabled = _env_flag("ASTA_TTS_FIRST_CLAUSE")
        self.output_latency = os.getenv("ASTA_TTS_OUTPUT_LATENCY", "low")
        self._stream = None
        self._stream_lock = threading.Lock()
        # Short, repeated phrases ("Okay, sir. Opening chrome.", "Opened
        # chrome.") are replayed from memory. This removes their synthesis
        # latency and keeps them off the GPU while the LLM is generating.
        try:
            self.cache_max_chars = max(0, int(os.getenv("ASTA_TTS_CACHE_CHARS", "80")))
            self.cache_size = max(0, int(os.getenv("ASTA_TTS_CACHE_SIZE", "64")))
        except (TypeError, ValueError):
            self.cache_max_chars, self.cache_size = 80, 64
        self._audio_cache = OrderedDict()
        self._cache_lock = threading.Lock()

        voice_start = time.perf_counter()
        self.pipeline.load_voice(self.voice)
        voice_load_time = time.perf_counter() - voice_start

        warmup_start = time.perf_counter()
        warmup_audio = 0
        try:
            for _, _, audio in self.pipeline(
                "Ready.",
                voice=self.voice,
                speed=self.speed,
                split_pattern=_SENTENCE_SPLIT,
            ):
                if audio is not None:
                    warmup_audio += int(self._to_numpy(audio).size)
        except Exception as exc:
            print(
                f"[Speech] Kokoro warmup warning: {type(exc).__name__}: {exc}",
                flush=True,
            )

        warmup_time = time.perf_counter() - warmup_start
        print(
            f"[Speech] Kokoro ready (voice={self.voice}, device={self.device}, "
            f"voice_load={voice_load_time:.3f}s, warmup={warmup_time:.3f}s, "
            f"warmup_audio={warmup_audio} samples)",
            flush=True,
        )

        if self.hindi_enabled and _env_flag("ASTA_TTS_HINDI_PRELOAD"):
            # Load the Hindi G2P and voice in the background so the first
            # Hindi reply does not pay ~3 s of setup.
            threading.Thread(
                target=self._preload_hindi, name="KokoroHindiPreload", daemon=True
            ).start()

    def _preload_hindi(self):
        pipeline, voice = self._pipeline_for("नमस्ते")
        if pipeline is self._hindi_pipeline:
            try:
                for _ in pipeline("नमस्ते।", voice=voice, speed=self.speed):
                    pass
            except Exception as exc:
                print(f"[Speech] Kokoro Hindi warmup warning: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Text handling
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(audio):
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    @staticmethod
    def is_hindi(text):
        letters = _LETTER.findall(str(text or ""))
        if not letters:
            return False
        devanagari = sum(1 for ch in letters if _DEVANAGARI.match(ch))
        return devanagari / len(letters) >= 0.3

    @classmethod
    def split_first_clause(cls, text, *, min_chars=20, max_chars=140):
        """Split ``text`` after its first clause when that clearly helps."""
        value = str(text or "").strip()
        if len(value) < min_chars * 2:
            return [value] if value else []
        for match in _FIRST_CLAUSE.finditer(value):
            cut = match.end()
            if cut > max_chars:
                break
            head, tail = value[:cut].strip(), value[cut:].strip()
            if len(head) >= min_chars and len(tail) >= min_chars:
                return [head, tail]
        return [value]

    def _pipeline_for(self, text):
        if not (self.hindi_enabled and self.is_hindi(text)) or self._hindi_failed:
            return self.pipeline, self.voice
        with self._hindi_lock:
            return self._load_hindi_pipeline()

    def _load_hindi_pipeline(self):
        if self._hindi_failed:
            return self.pipeline, self.voice
        if self._hindi_pipeline is None:
            try:
                start = time.perf_counter()
                # Share the already-loaded model; only the G2P differs.
                self._hindi_pipeline = KPipeline(
                    lang_code="h",
                    repo_id="hexgrad/Kokoro-82M",
                    model=getattr(self.pipeline, "model", True),
                )
                self._hindi_pipeline.load_voice(self.hindi_voice)
                print(
                    f"[Speech] Kokoro Hindi ready (voice={self.hindi_voice}, "
                    f"load={time.perf_counter() - start:.2f}s)",
                    flush=True,
                )
            except Exception as exc:
                self._hindi_failed = True
                print(
                    f"[Speech] Kokoro Hindi unavailable: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return self.pipeline, self.voice
        return self._hindi_pipeline, self.hindi_voice

    def iter_audio(self, text, should_continue=None, first_clause=True):
        """Yield float32 audio chunks for ``text`` as Kokoro produces them."""
        if not text:
            return
        pipeline, voice = self._pipeline_for(text)
        use_first_clause = bool(first_clause and self.first_clause_enabled)
        cache_key = None
        if self.cache_size and len(text) <= self.cache_max_chars:
            cache_key = (id(pipeline), voice, self.speed, use_first_clause, text)
            with self._cache_lock:
                cached = self._audio_cache.get(cache_key)
                if cached is not None:
                    self._audio_cache.move_to_end(cache_key)
            if cached is not None:
                print("[Speech] Kokoro cache hit (0 ms synth).", flush=True)
                for samples in cached:
                    if should_continue is not None and not should_continue():
                        return
                    yield samples
                return
        produced = [] if cache_key is not None else None
        pieces = (
            self.split_first_clause(text)
            if first_clause and self.first_clause_enabled
            else [text]
        )
        start = time.perf_counter()
        first_audio = None
        total = 0
        for piece in pieces:
            for _, _, audio in pipeline(
                piece,
                voice=voice,
                speed=self.speed,
                split_pattern=_SENTENCE_SPLIT,
            ):
                if should_continue is not None and not should_continue():
                    return
                if audio is None:
                    continue
                samples = self._to_numpy(audio)
                if samples.size == 0:
                    continue
                if first_audio is None:
                    first_audio = time.perf_counter() - start
                    print(f"[Speech] Kokoro TTFA: {first_audio:.3f}s", flush=True)
                total += int(samples.size)
                if produced is not None:
                    produced.append(samples)
                yield samples
        if produced:
            with self._cache_lock:
                self._audio_cache[cache_key] = tuple(produced)
                self._audio_cache.move_to_end(cache_key)
                while len(self._audio_cache) > self.cache_size:
                    self._audio_cache.popitem(last=False)
        print(
            f"[Speech] Kokoro synth: {time.perf_counter() - start:.2f}s | "
            f"audio: {total / self.SAMPLE_RATE:.2f}s",
            flush=True,
        )

    def synthesize(self, text):
        """Generate full audio (kept for callers that want synthesize-then-play)."""
        try:
            chunks = list(self.iter_audio(text))
        except Exception as exc:
            print(
                f"[Speech] Kokoro synthesis error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        return np.concatenate(chunks) if chunks else None

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    @staticmethod
    def _audio_level(audio_chunk):
        """Return a stable 0..1 envelope from an output audio chunk."""
        samples = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        return max(0.0, min(1.0, max(rms * 5.0, peak * 0.85)))

    def _ensure_stream(self):
        with self._stream_lock:
            if self._stream is None:
                start = time.perf_counter()
                self._stream = sd.OutputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    latency=self.output_latency,
                )
                print(
                    f"[Speech] Output stream opened ({time.perf_counter() - start:.3f}s)",
                    flush=True,
                )
            if not getattr(self._stream, "active", False):
                self._stream.start()
            return self._stream

    def write(self, audio, should_continue=None, on_level=None):
        """Write one audio chunk to the persistent stream. False if interrupted."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return True
        stream = self._ensure_stream()
        step = max(1, int(self.SAMPLE_RATE * self.WRITE_SECONDS))
        for index in range(0, samples.size, step):
            if should_continue is not None and not should_continue():
                self.abort()
                if on_level is not None:
                    on_level(0.0)
                return False
            piece = samples[index : index + step]
            stream.write(piece)
            if on_level is not None:
                on_level(self._audio_level(piece))
        return True

    def idle(self):
        """Let buffered audio finish, then pause the device until next use."""
        with self._stream_lock:
            if self._stream is not None and getattr(self._stream, "active", False):
                try:
                    self._stream.stop()
                except Exception as exc:
                    print(f"[Speech] Output stop warning: {exc}", flush=True)

    def abort(self):
        """Drop buffered audio immediately (barge-in)."""
        with self._stream_lock:
            if self._stream is not None and getattr(self._stream, "active", False):
                try:
                    self._stream.abort()
                except Exception as exc:
                    print(f"[Speech] Output abort warning: {exc}", flush=True)

    def close(self):
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def stream_play(self, text, should_continue=None, on_level=None, first_clause=True):
        """Synthesize on a helper thread and play chunks as they arrive."""
        chunks = queue.Queue()
        done = object()

        def produce():
            try:
                for chunk in self.iter_audio(
                    text, should_continue=should_continue, first_clause=first_clause
                ):
                    chunks.put(chunk)
            except Exception as exc:
                print(
                    f"[Speech] Kokoro synthesis error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                chunks.put(done)

        threading.Thread(target=produce, name="KokoroSynth", daemon=True).start()
        completed = True
        while True:
            chunk = chunks.get()
            if chunk is done:
                break
            if completed and not self.write(chunk, should_continue, on_level):
                completed = False
        if on_level is not None:
            on_level(0.0)
        return completed

    def play(self, audio, should_continue=None, on_level=None):
        """Play pre-generated audio and optionally report its live envelope."""
        if audio is None:
            if on_level is not None:
                on_level(0.0)
            return True
        try:
            completed = self.write(audio, should_continue, on_level)
            if completed:
                self.idle()
            if on_level is not None:
                on_level(0.0)
            if not completed:
                print("[Speech] Kokoro playback interrupted.", flush=True)
            return completed
        except Exception as exc:
            if on_level is not None:
                on_level(0.0)
            self.close()
            print(
                f"[Speech] Kokoro playback error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def speak(self, text):
        """Compatibility helper for callers that still want synthesize-then-play."""
        self.play(self.synthesize(text))
