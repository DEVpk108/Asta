import threading

from .applications import ApplicationManager
from .capability_discovery import CapabilityDiscovery
from .event_bus import EventBus
from .intent_router import IntentRouter
from .skill_manager import SkillManager
from .task_manager import TaskManager
from .workspace_manager import WorkspaceManager
from .tools import (
    ApprovalManager,
    AuthorityManager,
    AuthorityPolicy,
    ToolDispatcher,
    ToolRegistry,
)


class Kernel:
    """Central runtime for A.S.T.A. infrastructure."""

    def __init__(
        self,
        *,
        maximum_automatic_risk=None,
        authority_path=None,
    ):
        self.event_bus = EventBus()
        self.intent_router = IntentRouter()
        self.task_manager = TaskManager(event_bus=self.event_bus)
        self.workspace_manager = WorkspaceManager(event_bus=self.event_bus)
        self.application_manager = ApplicationManager()
        self.skill_manager = SkillManager(event_bus=self.event_bus)

        self.tool_registry = ToolRegistry()
        self.capability_discovery = CapabilityDiscovery(
            self.tool_registry,
            event_bus=self.event_bus,
        )
        self.approval_manager = ApprovalManager()

        policy = (
            AuthorityPolicy()
            if maximum_automatic_risk is None
            else AuthorityPolicy(
                maximum_automatic_risk=maximum_automatic_risk
            )
        )
        self.authority_manager = AuthorityManager(
            policy=policy,
            event_bus=self.event_bus,
            storage_path=authority_path,
        )

        self.tool_dispatcher = ToolDispatcher(
            registry=self.tool_registry,
            authority=self.authority_manager,
        )

        self.modules = []

        self._running = False
        self._stop_event = threading.Event()

    def create_task(self, goal, **kwargs):
        return self.task_manager.create(goal, **kwargs)

    @property
    def current_task(self):
        return self.task_manager.current()

    @property
    def workspace(self):
        return self.workspace_manager.snapshot()

    def register_module(self, module):
        if module not in self.modules:
            self.modules.append(module)
            print(f"[Kernel] Registered {module.name}")

    def register_tool(self, tool):
        self.tool_registry.register(tool)
        print(f"[Kernel] Registered tool {tool.definition.name}")

    def unregister_tool(self, name: str) -> bool:
        removed = self.tool_registry.unregister(name)
        if removed:
            print(f"[Kernel] Unregistered tool {name}")
        return removed

    def start(self):
        print("[Kernel] Starting...")

        for module in self.modules:
            print(f"[Kernel] Initializing {module.name}...")
            module.initialize()

        self._running = True
        self._stop_event.clear()
        self.event_bus.emit("kernel_ready")
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
