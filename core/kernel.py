import threading

from .event_bus import EventBus
from .intent_router import IntentRouter
from .tools import (
    ApprovalManager,
    AuthorityPolicy,
    ToolDispatcher,
    ToolRegistry,
)


class Kernel:
    """Central runtime for A.S.T.A. infrastructure."""

    def __init__(self, *, maximum_automatic_risk=None):
        self.event_bus = EventBus()
        self.intent_router = IntentRouter()

        self.tool_registry = ToolRegistry()
        self.approval_manager = ApprovalManager()

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
            print(f"[Kernel] Registered {module.name}")

    # ---------------------------------------------------------
    # Tool management
    # ---------------------------------------------------------

    def register_tool(self, tool):
        self.tool_registry.register(tool)
        print(f"[Kernel] Registered tool {tool.definition.name}")

    def unregister_tool(self, name: str) -> bool:
        removed = self.tool_registry.unregister(name)

        if removed:
            print(f"[Kernel] Unregistered tool {name}")

        return removed

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def start(self):
        print("[Kernel] Starting...")

        for module in self.modules:
            print(f"[Kernel] Initializing {module.name}...")
            module.initialize()

        self._running = True
        self._stop_event.clear()

        print("[Kernel] Running")

    def run(self):
        if not self._running:
            raise RuntimeError("Kernel must be started before run().")

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
        self.approval_manager.clear()

        for module in reversed(self.modules):
            try:
                module.shutdown()
            except Exception as exc:
                print(f"[Kernel] Error shutting down {module.name}: {exc}")

        print("[Kernel] Stopped")
