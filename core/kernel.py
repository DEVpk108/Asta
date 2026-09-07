import threading

from .event_bus import EventBus
from .intent_router import IntentRouter
from .tools import AuthorityPolicy, ToolDispatcher, ToolRegistry


class Kernel:
    """Central runtime for A.S.T.A. infrastructure.

    The kernel owns shared infrastructure and lifecycle. It does not
    interpret natural-language requests or implement tool-specific logic.
    """

    def __init__(self, *, maximum_automatic_risk=None):
        self.event_bus = EventBus()
        self.intent_router = IntentRouter()

        # Tool infrastructure is owned by the kernel so every module can
        # access the same registry/dispatcher without creating its own.
        self.tool_registry = ToolRegistry()

        policy = (
            AuthorityPolicy()
            if maximum_automatic_risk is None
            else AuthorityPolicy(
                maximum_automatic_risk=maximum_automatic_risk
            )
        )

        self.tool_dispatcher = ToolDispatcher(
            registry=self.tool_registry,
            policy=policy,
        )

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
    # Tool management
    # ---------------------------------------------------------

    def register_tool(self, tool):
        """Register a capability with the shared tool registry."""
        self.tool_registry.register(tool)
        print(
            f"[Kernel] Registered tool {tool.definition.name}"
        )

    def unregister_tool(self, name: str) -> bool:
        """Remove a capability from the shared tool registry."""
        removed = self.tool_registry.unregister(name)

        if removed:
            print(
                f"[Kernel] Unregistered tool {name}"
            )

        return removed

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
