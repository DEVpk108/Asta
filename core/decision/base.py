from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import DecisionSnapshot
from core.contracts.action import ActionDecision


class DecisionEngine(ABC):
    """Provider boundary for fast System-1 decisions."""

    name = "unknown"

    @abstractmethod
    def analyze(self, text: str) -> DecisionSnapshot:
        """Analyze one user request and return structured decisions."""
        raise NotImplementedError

    def decide_action(self, text: str, *, applications=None) -> ActionDecision:
        """Return a structured action decision when the provider supports it."""
        return ActionDecision(source=self.name)

    def warmup(self) -> bool:
        """Load provider resources during A.S.T.A. startup."""
        return True

    def shutdown(self) -> None:
        """Release provider resources during A.S.T.A. shutdown."""
        return None


class NullDecisionEngine(DecisionEngine):
    """No-op provider used when System-1 decisions are disabled."""

    name = "disabled"

    def analyze(self, text: str) -> DecisionSnapshot:
        return DecisionSnapshot(
            engine=self.name,
            model=None,
            input_text=str(text or "").strip(),
        )
