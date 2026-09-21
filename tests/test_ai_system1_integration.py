from core.contracts import DecisionSnapshot
from ai.ai_module import AIModule


class FakeEventBus:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **payload):
        self.events.append((event_type, payload))


class FakeDecisionEngine:
    def analyze(self, text):
        return DecisionSnapshot(
            engine="laya",
            model="english",
            input_text=text,
            decisions={
                "intent": {"choice": "question", "confidence": 0.9},
            },
            routing={"model": "english"},
            latency_ms=12.5,
        )


def _build_module(decision_engine):
    module = object.__new__(AIModule)
    event_bus = FakeEventBus()
    module.kernel = type(
        "KernelStub",
        (),
        {
            "decision_engine": decision_engine,
            "event_bus": event_bus,
        },
    )()
    module.event_bus = event_bus
    return module


def test_ai_module_emits_system1_decision_result():
    module = _build_module(FakeDecisionEngine())

    module._run_system1_decision("what is a transformer?")

    assert module.kernel.event_bus.events == [
        (
            "decision_result",
            {
                "decision": module.kernel.event_bus.events[0][1]["decision"],
            },
        )
    ]
    decision = module.kernel.event_bus.events[0][1]["decision"]
    assert decision.engine == "laya"
    assert decision.choice("intent") == "question"


def test_ai_module_system1_failure_does_not_raise():
    class FailingEngine:
        def analyze(self, text):
            raise RuntimeError("simulated decision failure")

    module = _build_module(FailingEngine())

    module._run_system1_decision("hello")
    assert module.kernel.event_bus.events == []
