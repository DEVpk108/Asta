"""
A.S.T.A. Cognitive OS core infrastructure.

The core package provides runtime orchestration, tasks, workspaces,
capability discovery, skills, context, decision providers, and host application
discovery.
"""

from .applications import ApplicationManager, ApplicationRecord, ApplicationResolutionError
from .capability_discovery import CapabilityDiscovery
from .context_builder import ContextBuilder, ContextSnapshot
from .decision import (
    DecisionEngine,
    DecisionSnapshot,
    LayaDecisionEngine,
    NullDecisionEngine,
    create_decision_engine,
)
from .event_bus import EventBus
from .kernel import Kernel
from .module import Module
from .planner import Planner, PlanningError
from .skill_manager import SkillManager
from .task_manager import TaskManager
from .workspace_manager import WorkspaceManager

__all__ = [
    "ApplicationManager",
    "ApplicationRecord",
    "ApplicationResolutionError",
    "CapabilityDiscovery",
    "ContextBuilder",
    "ContextSnapshot",
    "DecisionEngine",
    "DecisionSnapshot",
    "LayaDecisionEngine",
    "NullDecisionEngine",
    "create_decision_engine",
    "EventBus",
    "Kernel",
    "Module",
    "Planner",
    "PlanningError",
    "SkillManager",
    "TaskManager",
    "WorkspaceManager",
]
