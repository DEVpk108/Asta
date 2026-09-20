from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Reusable problem-solving instructions that guide model reasoning."""

    name: str
    description: str
    instructions: str
    triggers: tuple[str, ...] = ()
    provider: str = "native"
    priority: int = 0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "triggers": list(self.triggers),
            "provider": self.provider,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }
