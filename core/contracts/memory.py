from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """
    A single piece of persistent ASTA memory.
    """

    id: str
    content: str
    memory_type: str

    importance: float

    source: str

    created_at: datetime

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(
                "importance must be between 0.0 and 1.0"
            )


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """
    Query sent to the memory subsystem.
    """

    text: str
    limit: int = 5

    memory_types: tuple[str, ...] = ()

    min_importance: float = 0.0

    def __post_init__(self):
        if self.limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        if not 0.0 <= self.min_importance <= 1.0:
            raise ValueError(
                "min_importance must be between 0.0 and 1.0"
            )