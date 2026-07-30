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
- Plugin Manager

=========================================================
"""

from .event_bus import EventBus
from .module import Module

__all__ = [
    "EventBus","Module",
]