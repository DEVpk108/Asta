import threading

from core.module import Module


class TextInputModule(Module):
    """Simple terminal input adapter that feeds text into ASTA's event bus."""

    def __init__(self, kernel):
        super().__init__(
            name="TextInput",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self._running = False
        self._thread = None

    def initialize(self):
        print("[Text] Initializing...")

        self._running = True
        self._thread = threading.Thread(
            target=self._input_loop,
            name="TextInputLoop",
            daemon=True,
        )
        self._thread.start()

        print("[Text] Ready")
        print("[Text] Type a message and press Enter.", flush=True)

    def shutdown(self):
        self._running = False
        self._thread = None

    def _input_loop(self):
        while self._running:
            try:
                text = input("[Text] You: ")
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as exc:
                print(
                    f"[Text] Input error: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            text = text.strip()

            if not text:
                continue

            self.event_bus.emit("user_message", text=text)
