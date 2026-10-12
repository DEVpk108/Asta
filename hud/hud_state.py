from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HUDState:
    """Presentation state exposed by A.S.T.A. to the HUD."""

    mode: str = "idle"
    intensity: str = "low"
    status: str = "IDLE"
    progress: Optional[float] = None
    activity: Optional[str] = None
    model_profile: Optional[str] = None
    model_status: Optional[str] = None
    model_name: Optional[str] = None
    model_services: dict = field(default_factory=dict)
    background_task: Optional[dict] = None
    project_steps: list = field(default_factory=list)
    approvals: list = field(default_factory=list)
    model_mode: Optional[str] = None
