"""
A.S.T.A. Cognitive OS core infrastructure.

The core package provides runtime orchestration, tasks, workspaces,
capability discovery, skills, context, decision providers, and host application
discovery.
"""

from .agent import AgentBrain, AgentBrainError, AgentBelief, AgentObservation, AgentState
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
from .notes_manager import NotesManager
from .planner import Planner, PlanningError
from .skill_manager import SkillManager
from .task_manager import TaskManager
from .workspace_manager import WorkspaceManager

__all__ = [
    "AgentBrain",
    "AgentBrainError",
    "AgentBelief",
    "AgentObservation",
    "AgentState",
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
    "NotesManager",
    "Planner",
    "PlanningError",
    "SkillManager",
    "TaskManager",
    "WorkspaceManager",
]
