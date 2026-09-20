from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Description of a capability that A.S.T.A. can discover and use."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    requires_confirmation: bool = False
    provider: str = "native"
    loaded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_definition(
        cls,
        definition,
        *,
        provider: str = "native",
        loaded: bool = True,
    ) -> "CapabilityDescriptor":
        return cls(
            name=definition.name,
            description=definition.description,
            input_schema=dict(definition.input_schema or {}),
            risk_level=definition.risk_level,
            requires_confirmation=definition.requires_confirmation,
            provider=provider,
            loaded=loaded,
            metadata=dict(definition.metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "risk_level": self.risk_level,
            "requires_confirmation": self.requires_confirmation,
            "provider": self.provider,
            "loaded": self.loaded,
            "metadata": dict(self.metadata),
        }
