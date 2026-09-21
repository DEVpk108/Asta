from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import DecisionSnapshot


class DecisionEngine(ABC):
    """Provider boundary for fast System-1 decisions."""

    name = "unknown"

    @abstractmethod
    def analyze(self, text: str) -> DecisionSnapshot:
        """Analyze one user request and return structured decisions."""
        raise NotImplementedError


class NullDecisionEngine(DecisionEngine):
    """No-op provider used when System-1 decisions are disabled."""

    name = "disabled"

    def analyze(self, text: str) -> DecisionSnapshot:
        return DecisionSnapshot(
            engine=self.name,
            model=None,
            input_text=str(text or "").strip(),
        )
