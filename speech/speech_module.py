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

        # Local-first TTS backend. Kokoro uses the local GPU when available.
        # Its model/inference path is warmed during construction.
        self.engine = KokoroEngine()

        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self.coalesce_window = 0.10

    def initialize(self):
        print("[Speech] Initializing...", flush=True)
        self._running = True
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)
        self._thread = threading.Thread(
            target=self._speech_loop,
            name="SpeechWorker",
            daemon=True,
        )
        self._thread.start()
        print("[Speech] Ready", flush=True)

    def shutdown(self):
        print("[Speech] Shutting down...", flush=True)
        self._running = False
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)
        self._queue.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
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

    def _speech_loop(self):
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

                print(f"[Speech] {text}", flush=True)
                try:
                    self.engine.speak(text)
                except Exception as exc:
                    print(
                        f"[Speech] Error: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    self._queue.task_done()
            except Exception as exc:
                print(
                    f"[Speech] Worker error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
