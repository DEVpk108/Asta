from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    CONVERSATION = "conversation"
    COMMAND = "command"
    MEMORY = "memory"
    TASK = "task"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IntentResult:
    """
    Result produced by the Intent Engine.

    The Intent Engine describes what the user wants.
    It does not execute the request or select a model.
    """

    intent: IntentType
    confidence: float
    normalized_text: str

    entities: dict[str, Any] = field(
        default_factory=dict
    )

    requires_tools: bool = False
    requires_memory: bool = False
    requires_reasoning: bool = False

    classifier: str = "rules"

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )