import sys
import types

import pytest

from core.contracts import DecisionSnapshot
from core.decision import (
    LayaDecisionEngine,
    NullDecisionEngine,
    create_decision_engine,
)


def test_decision_snapshot_helpers():
    snapshot = DecisionSnapshot(
        engine="test",
        model="model-a",
        input_text="hello",
        decisions={
            "intent": {"choice": "conversation", "confidence": 0.9},
            "needs_tools": {"noul": 0.1},
            "difficulty": {"score": 1.5},
        },
    )

    assert snapshot.choice("intent") == "conversation"
    assert snapshot.probability("needs_tools") == 0.1
    assert snapshot.score("difficulty") == 1.5
    assert snapshot.to_dict()["engine"] == "test"


def test_default_decision_engine_is_disabled(monkeypatch):
    monkeypatch.delenv("ASTA_DECISION_ENGINE", raising=False)
    engine = create_decision_engine()

    assert isinstance(engine, NullDecisionEngine)
    assert engine.analyze("hello").engine == "disabled"


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("ASTA_DECISION_ENGINE", "unknown")

    with pytest.raises(ValueError):
        create_decision_engine()


def test_laya_engine_maps_router_result(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions):
            assert state == {"user_request": "open spotify"}
            assert "intent" in questions
            assert "needs_tools" in questions
            return {
                "model": "laya-rl-agent",
                "answers": {
                    "intent": {
                        "type": "choice",
                        "choice": "command",
                        "confidence": 0.95,
                    },
                    "needs_tools": {
                        "type": "noul",
                        "noul": 0.99,
                        "confidence": 0.99,
                    },
                },
                "routing": {
                    "model": "english",
                    "reason": "English Latin text",
                },
            }

    monkeypatch.setitem(
        sys.modules,
        "laya",
        types.SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(
        device="cpu",
        preload=False,
        max_loaded=1,
    )
    snapshot = engine.analyze("open spotify")

    assert snapshot.engine == "laya"
    assert snapshot.model == "english"
    assert snapshot.choice("intent") == "command"
    assert snapshot.probability("needs_tools") == 0.99
    assert snapshot.routing["reason"] == "English Latin text"
    assert snapshot.latency_ms is not None


def test_laya_engine_reports_missing_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "laya", raising=False)

    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("missing laya")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    engine = LayaDecisionEngine()

    with pytest.raises(RuntimeError, match="laya.*not installed"):
        engine.analyze("hello")
