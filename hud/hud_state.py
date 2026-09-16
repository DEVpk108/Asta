from dataclasses import dataclass
from typing import Optional


@dataclass
class HUDState:
    """Presentation state exposed by A.S.T.A. to the HUD."""

    mode: str = "idle"
    intensity: str = "low"
    status: str = "IDLE"
    progress: Optional[float] = None
    activity: Optional[str] = None
