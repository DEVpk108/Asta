import queue
import threading

from core.module import Module
from .polly_engine import PollyEngine


class SpeechModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = PollyEngine()

        self._queue = queue.Queue()
        self._running = False
        self._thread = None

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def initialize(self):
        print("[Speech] Initializing...")

        self._running = True

        self.event_bus.subscribe(
            "assistant_sentence",
            self.on_assistant_sentence,
        )

        self._thread = threading.Thread(
            target=self._speech_loop,
            name="SpeechWorker",
            daemon=True,
        )

        self._thread.start()

        print("[Speech] Ready")

    def shutdown(self):
        print("[Speech] Shutting down...")

        self._running = False

        self.event_bus.unsubscribe(
            "assistant_sentence",
            self.on_assistant_sentence,
        )

        # Wake the worker if it is waiting.
        self._queue.put(None)

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):
            self._thread.join(timeout=2.0)

        self._thread = None

        print("[Speech] Stopped")

    # ---------------------------------------------------------
    # Event handler
    # ---------------------------------------------------------

    def on_assistant_sentence(self, text):
        if not text:
            return

        # Do NOT call Polly here.
        # EventBus.emit() is synchronous.
        self._queue.put(text)

    # ---------------------------------------------------------
    # Speech worker
    # ---------------------------------------------------------

    def _speech_loop(self):

        while self._running:

            try:
                text = self._queue.get()

                if text is None:
                    break

                print(
                    f"[Speech] {text}",
                    flush=True,
                )

                try:
                    self.engine.speak(text)

                except Exception as exc:
                    print(
                        f"[Speech] Error: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

                finally:
                    self._queue.task_done()

            except Exception as exc:
                print(
                    f"[Speech] Worker error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )