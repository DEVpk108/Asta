"""
A.S.T.A. Cognitive OS core infrastructure.

The core package provides runtime orchestration, tasks, workspaces,
capability discovery, skills, context, and host application discovery.
"""

from .applications import ApplicationManager, ApplicationRecord, ApplicationResolutionError
from .capability_discovery import CapabilityDiscovery
from .context_builder import ContextBuilder, ContextSnapshot
from .event_bus import EventBus
from .kernel import Kernel
from .module import Module
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
    "EventBus",
    "Kernel",
    "Module",
    "SkillManager",
    "TaskManager",
    "WorkspaceManager",
]
