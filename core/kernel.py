import threading

from .applications import ApplicationManager
from .autonomy import (
    CapabilitySetupManager,
    DiagnosisEngine,
    ReplanEngine,
    RecoveryManager,
    VerificationEngine,
)
from .capability_discovery import CapabilityDiscovery
from .decision import create_decision_engine
from .event_bus import EventBus
from .intent_router import IntentRouter
from .notes_manager import NotesManager
from .media import MediaManager
from .planner import Planner
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
        self.media_manager = MediaManager()
        self.intent_router = IntentRouter(
            media_providers=self.media_manager.providers(),
        )
        self.decision_engine = create_decision_engine()
        self.task_manager = TaskManager(event_bus=self.event_bus)
        self.recovery_manager = RecoveryManager()
        self.diagnosis_engine = DiagnosisEngine(self.decision_engine)
        self.replan_engine = ReplanEngine(self.decision_engine)
        self.verification_engine = VerificationEngine(self.media_manager)
        self.workspace_manager = WorkspaceManager(event_bus=self.event_bus)
        self.application_manager = ApplicationManager()
        self.skill_manager = SkillManager(event_bus=self.event_bus)
        self.notes_manager = NotesManager()

        self.tool_registry = ToolRegistry()
        self.planner = Planner(
            self.tool_registry,
            media_manager=self.media_manager,
            application_manager=self.application_manager,
        )
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
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_owner = None
        self._shutdown_complete = threading.Event()

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

        with self._shutdown_lock:
            self._shutdown_started = False
            self._shutdown_owner = None
            self._shutdown_complete.clear()

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
        current_thread = threading.get_ident()

        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                return

            if self._shutdown_started:
                if self._shutdown_owner == current_thread:
                    return
                owner = False
            else:
                self._shutdown_started = True
                self._shutdown_owner = current_thread
                self._running = False
                self._stop_event.set()
                owner = True

        if not owner:
            # Another thread owns the shutdown sequence (for example the HUD
            # transport worker). The main runtime must stay alive until that
            # sequence has completed instead of exiting early and leaving
            # managed child processes such as llama-server behind.
            self._shutdown_complete.wait()
            return

        try:
            print("[Kernel] Shutting down...")

            self.approval_manager.clear()

            for module in reversed(self.modules):
                try:
                    module.shutdown()
                except Exception as exc:
                    print(f"[Kernel] Error shutting down {module.name}: {exc}")

            print("[Kernel] Stopped")
        finally:
            with self._shutdown_lock:
                self._shutdown_owner = None
                self._shutdown_complete.set()
