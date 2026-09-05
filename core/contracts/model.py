from dataclasses import dataclass, field
from typing import Any

from .intent import IntentResult
from .memory import MemoryRecord


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """
    Normalized request sent to the Model Orchestrator.
    """

    prompt: str

    intent: IntentResult

    memories: tuple[MemoryRecord, ...] = ()

    tools: tuple[str, ...] = ()

    max_latency_ms: int | None = None

    require_local: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """
    Normalized result returned by a model provider.
    """

    text: str

    model: str

    provider: str

    input_tokens: int = 0
    output_tokens: int = 0

    ttft_seconds: float = 0.0
    tokens_per_second: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )