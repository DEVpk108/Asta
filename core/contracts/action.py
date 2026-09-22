from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    NONE = "none"
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    OPEN_URL = "open_url"
    WEB_SEARCH = "web_search"
    OPEN_FOLDER = "open_folder"
    TYPE_TEXT = "type_text"
    KEYPRESS = "keypress"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    MEDIA = "media"
    VOLUME = "volume"
    SYSTEM = "system"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """System-1 action decision selected from a finite A.S.T.A. action space."""

    action: ActionType = ActionType.NONE
    confidence: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)
    addressed: float = 0.0
    command_complete: bool = True
    compound: bool = False
    source: str = "unknown"
    model: str | None = None
    latency_ms: float | None = None
    raw_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not 0.0 <= float(self.addressed) <= 1.0:
            raise ValueError("addressed must be between 0.0 and 1.0")

    @property
    def is_actionable(self) -> bool:
        return (
            self.action is not ActionType.NONE
            and self.confidence >= 0.50
            and self.addressed >= 0.50
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": float(self.confidence),
            "arguments": dict(self.arguments),
            "addressed": float(self.addressed),
            "command_complete": bool(self.command_complete),
            "compound": bool(self.compound),
            "source": self.source,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "raw_decisions": {
                key: dict(value)
                for key, value in self.raw_decisions.items()
            },
        }
