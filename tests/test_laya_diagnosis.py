from types import SimpleNamespace

from core.contracts.action import ActionType
from core.decision.laya_engine import LayaDecisionEngine


def test_laya_engine_diagnoses_failure_with_bounded_questions(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions, model=None):
            assert state["goal"] == "Play Hanuman Chalisa on Spotify"
            assert state["failed_tool"] == "media.control"
            assert state["error"] == "no active Spotify device"
            assert "category" in questions
            assert "recommended_action" in questions
            assert "requires_user" in questions
            assert model == "multilingual"
            return {
                "model": "laya-multilingual",
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "state_mismatch",
                        "confidence": 0.92,
                    },
                    "recommended_action": {
                        "type": "choice",
                        "choice": "replan",
                        "confidence": 0.89,
                    },
                    "requires_user": {
                        "type": "noul",
                        "noul": 0.08,
                        "confidence": 0.94,
                    },
                },
                "routing": {
                    "model": "multilingual",
                    "reason": "explicit model='multilingual'",
                },
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "laya",
        SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(model="multilingual")
    diagnosis = engine.diagnose_failure(
        {
            "task_id": "task-1",
            "goal": "Play Hanuman Chalisa on Spotify",
            "failed_tool": "media.control",
            "step_id": "step-1",
            "attempt": 2,
            "error": "no active Spotify device",
            "output": "",
        }
    )

    assert diagnosis.category.value == "state_mismatch"
    assert diagnosis.recommended_action == "replan"
    assert diagnosis.requires_user is False
    assert diagnosis.confidence == 0.89
    assert diagnosis.source == "laya"
    assert diagnosis.model == "multilingual"
    assert diagnosis.latency_ms is not None
    assert diagnosis.task_id == "task-1"


def test_laya_engine_forces_user_action_for_user_boundary(monkeypatch):
    class FakeRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def predict(self, state, questions, model=None):
            return {
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "authentication",
                        "confidence": 0.93,
                    },
                    "recommended_action": {
                        "type": "choice",
                        "choice": "retry",
                        "confidence": 0.91,
                    },
                    "requires_user": {
                        "type": "noul",
                        "noul": 0.96,
                        "confidence": 0.98,
                    },
                },
                "routing": {"model": "multilingual"},
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "laya",
        SimpleNamespace(Router=FakeRouter),
    )

    engine = LayaDecisionEngine(model="multilingual")
    diagnosis = engine.diagnose_failure(
        {
            "task_id": "task-1",
            "goal": "Connect Spotify",
            "failed_tool": "media.control",
            "step_id": "step-1",
            "attempt": 2,
            "error": "login required",
            "output": "",
        }
    )

    assert diagnosis.category.value == "authentication"
    assert diagnosis.requires_user is True
    assert diagnosis.recommended_action == "wait_for_user"
