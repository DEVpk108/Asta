import queue
import threading
import time

from core.module import Module
from .kokoro_engine import KokoroEngine


class SpeechModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        # Local-first TTS backend. Synthesis and playback are separated so the
        # next sentence can be prepared while the previous sentence is playing.
        self.engine = KokoroEngine()

        self._queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._running = False
        self._synth_thread = None
        self._play_thread = None
        self.coalesce_window = 0.08

    def initialize(self):
        print("[Speech] Initializing...", flush=True)
        self._running = True
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)

        self._synth_thread = threading.Thread(
            target=self._synthesis_loop,
            name="SpeechSynthWorker",
            daemon=True,
        )
        self._play_thread = threading.Thread(
            target=self._playback_loop,
            name="SpeechPlaybackWorker",
            daemon=True,
        )
        self._synth_thread.start()
        self._play_thread.start()
        print("[Speech] Ready", flush=True)

    def shutdown(self):
        print("[Speech] Shutting down...", flush=True)
        self._running = False
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)

        self._queue.put(None)
        self._audio_queue.put(None)

        if self._synth_thread is not None and self._synth_thread.is_alive():
            self._synth_thread.join(timeout=2.0)
        if self._play_thread is not None and self._play_thread.is_alive():
            self._play_thread.join(timeout=2.0)

        self._synth_thread = None
        self._play_thread = None
        print("[Speech] Stopped", flush=True)

    def on_assistant_sentence(self, text):
        if text:
            self._queue.put(text)

    def _get_coalesced_text(self, first_text):
        """Combine chunks already arriving, with a tiny debounce window."""
        parts = [first_text]
        deadline = time.monotonic() + self.coalesce_window

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_text = self._queue.get(timeout=remaining)
            except queue.Empty:
                break

            if next_text is None:
                self._queue.task_done()
                self._queue.put(None)
                break

            parts.append(next_text)
            self._queue.task_done()

        return " ".join(part.strip() for part in parts if part and part.strip())

    def _synthesis_loop(self):
        while self._running:
            try:
                text = self._queue.get()
                if text is None:
                    self._queue.task_done()
                    break

                text = self._get_coalesced_text(text)
                if not text:
                    self._queue.task_done()
                    continue

                print(f"[Speech] Synthesizing: {text}", flush=True)
                try:
                    audio = self.engine.synthesize(text)
                    if audio is not None:
                        self._audio_queue.put(audio)
                except Exception as exc:
                    print(
                        f"[Speech] Synthesis worker error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    self._queue.task_done()
            except Exception as exc:
                print(
                    f"[Speech] Worker error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _playback_loop(self):
        while self._running:
            try:
                audio = self._audio_queue.get()
                if audio is None:
                    self._audio_queue.task_done()
                    break

                try:
                    self.engine.play(audio)
                except Exception as exc:
                    print(
                        f"[Speech] Playback worker error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    self._audio_queue.task_done()
            except Exception as exc:
                print(
                    f"[Speech] Playback loop error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
