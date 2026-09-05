from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """
    Structured request to execute a tool.
    """

    tool: str

    arguments: dict[str, Any]

    request_id: str

    timeout_seconds: float = 30.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Structured result returned by a tool.
    """

    success: bool

    tool: str

    output: Any = None

    error: str | None = None

    duration_seconds: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """
    Metadata describing a tool available to ASTA.
    """

    name: str

    description: str

    input_schema: dict[str, Any]

    risk_level: str = "low"

    requires_confirmation: bool = False

    timeout_seconds: float = 30.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )