from __future__ import annotations

import os

from .base import DecisionEngine, NullDecisionEngine
from .laya_engine import LayaDecisionEngine


def create_decision_engine(provider: str | None = None) -> DecisionEngine:
    value = str(
        provider or os.getenv("ASTA_DECISION_ENGINE", "disabled")
    ).strip().lower()

    if value in {"disabled", "none", "off"}:
        return NullDecisionEngine()

    if value in {"laya", "laya_system1"}:
        return LayaDecisionEngine()

    raise ValueError(
        f"Unknown decision engine '{value}'. "
        "Supported values: disabled, laya."
    )
