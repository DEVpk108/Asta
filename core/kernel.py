import threading

from .event_bus import EventBus


class Kernel:

    def __init__(self):
        self.event_bus = EventBus()
        self.modules = []
        self._running = False
        self._stop_event = threading.Event()

    # ---------------------------------------------------------
    # Module management
    # ---------------------------------------------------------

    def register_module(self, module):
        if module not in self.modules:
            self.modules.append(module)
            print(
                f"[Kernel] Registered {module.name}"
            )

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def start(self):
        print("[Kernel] Starting...")

        for module in self.modules:
            print(
                f"[Kernel] Initializing "
                f"{module.name}..."
            )
            module.initialize()

        self._running = True
        self._stop_event.clear()

        print("[Kernel] Running")

    def run(self):
        if not self._running:
            raise RuntimeError(
               "Kernel must be started before run()."
            )

        try:
            while self._running:
                self._stop_event.wait(0.5)

        except KeyboardInterrupt:
            print("\n[Kernel] Keyboard interrupt")

        finally:
            self.shutdown()

    def shutdown(self):
        if not self._running:
            return

        print("[Kernel] Shutting down...")

        self._running = False
        self._stop_event.set()

        for module in reversed(self.modules):
            try:
                module.shutdown()
            except Exception as exc:
                print(
                    f"[Kernel] Error shutting down "
                    f"{module.name}: {exc}"
                )

        print("[Kernel] Stopped")