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


def test_ai_module_emits_system1_decision_result():
    module = object.__new__(AIModule)
    module.kernel = type(
        "KernelStub",
        (),
        {
            "decision_engine": FakeDecisionEngine(),
            "event_bus": FakeEventBus(),
        },
    )()

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

    module = object.__new__(AIModule)
    module.kernel = type(
        "KernelStub",
        (),
        {
            "decision_engine": FailingEngine(),
            "event_bus": FakeEventBus(),
        },
    )()

    module._run_system1_decision("hello")
    assert module.kernel.event_bus.events == []
