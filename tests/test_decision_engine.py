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


def test_laya_engine_uses_multilingual_model_by_default(monkeypatch):
    monkeypatch.delenv("ASTA_LAYA_MODEL", raising=False)

    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions, model=None):
            assert state == {"user_request": "open spotify"}
            assert "intent" in questions
            assert "needs_tools" in questions
            assert model == "multilingual"
            return {
                "model": "laya-multilingual",
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
                    "model": "multilingual",
                    "reason": "explicit model='multilingual'",
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

    assert engine.model == "multilingual"
    assert engine._router.kwargs["default"] == "multilingual"
    assert engine._router.kwargs["preload"] is False
    assert snapshot.engine == "laya"
    assert snapshot.model == "multilingual"
    assert snapshot.choice("intent") == "command"
    assert snapshot.probability("needs_tools") == 0.99
    assert snapshot.routing["reason"] == "explicit model='multilingual'"
    assert snapshot.latency_ms is not None


def test_laya_engine_model_can_be_overridden(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions, model=None):
            assert model == "english"
            return {
                "answers": {},
                "routing": {
                    "model": "english",
                    "reason": "explicit model='english'",
                },
            }

    monkeypatch.setitem(
        sys.modules,
        "laya",
        types.SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(model="english")
    snapshot = engine.analyze("hello")

    assert engine.model == "english"
    assert snapshot.model == "english"


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


def test_laya_engine_warmup_preloads_selected_model(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.preloaded = []

        def preload(self, names):
            self.preloaded.append(list(names))

    monkeypatch.setitem(
        sys.modules,
        "laya",
        types.SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(
        model="multilingual",
        preload=True,
    )

    assert engine.warmup() is True
    assert engine.warmed is True
    assert engine._router.preloaded == [["multilingual"]]


def test_laya_engine_shutdown_unloads_router(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.unload_calls = 0

        def load(self, name):
            return object()

        def unload(self):
            self.unload_calls += 1

    monkeypatch.setitem(
        sys.modules,
        "laya",
        types.SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(model="multilingual", preload=False)

    assert engine.preload is False
    assert engine.warmup() is True
    router = engine._router

    engine.shutdown()

    assert router.unload_calls == 1
    assert engine._router is None
    assert engine.warmed is False



def test_laya_engine_decide_action_selects_structured_app_action(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions, model=None):
            assert state == {
                "user_request": "close it",
                "applications": ["Spotify", "Visual Studio Code"],
            }
            assert "action" in questions
            assert "target_app" in questions
            assert "command_complete" in questions
            assert model == "multilingual"
            return {
                "model": "laya-multilingual",
                "answers": {
                    "action": {
                        "type": "choice",
                        "choice": "close_app",
                        "confidence": 0.96,
                    },
                    "addressed": {
                        "type": "noul",
                        "noul": 0.99,
                        "confidence": 0.99,
                    },
                    "compound": {
                        "type": "noul",
                        "noul": 0.04,
                        "confidence": 0.96,
                    },
                    "command_complete": {
                        "type": "noul",
                        "noul": 0.98,
                        "confidence": 0.99,
                    },
                    "target_app": {
                        "type": "choice",
                        "choice": "Spotify",
                        "confidence": 0.94,
                    },
                },
                "routing": {
                    "model": "multilingual",
                    "reason": "explicit model='multilingual'",
                },
            }

    monkeypatch.setitem(
        sys.modules,
        "laya",
        types.SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(model="multilingual")
    decision = engine.decide_action(
        "close it",
        applications=["Spotify", "Visual Studio Code"],
    )

    assert decision.action.value == "close_app"
    assert decision.arguments == {"target_app": "Spotify"}
    assert decision.confidence == pytest.approx(0.94)
    assert decision.addressed == pytest.approx(0.99)
    assert decision.command_complete is True
    assert decision.compound is False
    assert decision.source == "laya"
    assert decision.model == "multilingual"
    assert decision.latency_ms is not None
