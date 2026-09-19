"""
=========================================================
A.S.T.A. Cognitive OS
Core Package
---------------------------------------------------------
Contains the core infrastructure of A.S.T.A., including:

- Event Bus
- Kernel
- Module System
- Task Manager
- Workspace Manager
- Capability Discovery
- Skill Manager
- Context Builder
- Plugin Manager

=========================================================
"""

from .capability_discovery import CapabilityDiscovery
from .context_builder import ContextBuilder, ContextSnapshot
from .event_bus import EventBus
from .kernel import Kernel
from .module import Module
from .skill_manager import SkillManager
from .task_manager import TaskManager
from .workspace_manager import WorkspaceManager

__all__ = [
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
