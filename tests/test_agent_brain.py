from core import Kernel
from core.agent import AgentBrain
from core.contracts import IntentResult, IntentType, ToolDefinition
from core.tools import Tool


class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open a target for tests.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request):
        raise AssertionError("agent brain tests do not execute tools")


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def generate_response(self, text, **kwargs):
        self.prompts.append(text)
        return self.response


def _intent():
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="open calculator",
        entities={"action": "open", "target": "calculator"},
        requires_tools=True,
        classifier="rules",
    )


def test_agent_brain_builds_structured_plan_from_local_model():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    provider = FakeProvider(
        """{
            "goal_summary": "Open the calculator application.",
            "success_conditions": ["Calculator is open."],
            "rationale": "The requested goal requires opening the target application.",
            "uncertainty": 0.1,
            "steps": [
                {"action": "open", "target": "calculator"}
            ]
        }"""
    )
    brain = AgentBrain(
        kernel,
        provider=provider,
        enabled=True,
    )

    proposal = brain.plan(
        "open calculator",
        intent=_intent(),
    )

    assert proposal.goal_summary == "Open the calculator application."
    assert proposal.success_conditions == ("Calculator is open.",)
    assert proposal.steps == ({"action": "open", "target": "calculator"},)
    assert proposal.uncertainty == 0.1
    assert provider.prompts


def test_agent_brain_rejects_unsupported_actions():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    provider = FakeProvider(
        """{
            "goal_summary": "Do something unsafe.",
            "success_conditions": ["Done."],
            "rationale": "Test.",
            "uncertainty": 0.5,
            "steps": [
                {"action": "invented_action", "target": "calculator"}
            ]
        }"""
    )
    brain = AgentBrain(
        kernel,
        provider=provider,
        enabled=True,
    )

    try:
        brain.plan("do something", intent=_intent())
    except RuntimeError as exc:
        assert "unsupported action" in str(exc)
    else:
        raise AssertionError("Expected unsupported action rejection")
