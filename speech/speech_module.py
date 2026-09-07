import queue
import threading

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
        self.engine = KokoroEngine()

        self._queue = queue.Queue()
        self._running = False
        self._thread = None

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

    def _speech_loop(self):
        while self._running:
            try:
                text = self._queue.get()
                if text is None:
                    break
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
