from .event_bus import EventBus


class Kernel:

    def __init__(self):
        self.event_bus = EventBus()
        self.modules = []
        self._running = False

    # ---------------------------------------------------------
    # Module Management
    # ---------------------------------------------------------

    def register_module(self, module):
        if module not in self.modules:
            self.modules.append(module)
            print(f"[Kernel] Registered {module.name}")

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def start(self):

        print("[Kernel] Starting...")

        for module in self.modules:
            print(f"[Kernel] Initializing {module.name}...")
            module.initialize()

        self._running = True

        print("[Kernel] Running")

    def run(self):

        self._running = True

        try:
            while self._running:
                pass

        except KeyboardInterrupt:
            print("\n[Kernel] Keyboard interrupt")

        finally:
            self.shutdown()

    def shutdown(self):

        if not self._running:
            return

        print("[Kernel] Shutting down...")

        self._running = False

        for module in reversed(self.modules):
            try:
                module.shutdown()
            except Exception as exc:
                print(
                    f"[Kernel] Error shutting down "
                    f"{module.name}: {exc}"
                )

        print("[Kernel] Stopped")