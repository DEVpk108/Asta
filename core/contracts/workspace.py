from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(slots=True)
class WorkspaceState:
    """Structured state describing the environment A.S.T.A. is working in."""

    project_name: str = ""
    project_path: str = ""
    repository: str = ""
    branch: str = ""
    active_files: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached snapshot safe to expose to other runtime layers."""
        return {
            "project_name": self.project_name,
            "project_path": self.project_path,
            "repository": self.repository,
            "branch": self.branch,
            "active_files": list(self.active_files),
            "recent_files": list(self.recent_files),
            "environment": dict(self.environment),
            "hardware": dict(self.hardware),
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at,
        }
