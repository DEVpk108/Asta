from core import Kernel, Planner
from core.agent import AgentPlanProposal
from core.contracts import IntentResult, IntentType, ToolDefinition
from core.tools import Tool


class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open something for tests.",
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
        raise AssertionError("planner tests must not execute tools")


class FakeBrain:
    enabled = True

    def __init__(self):
        self.calls = []

    def plan(self, goal, *, intent):
        self.calls.append((goal, intent.entities))
        return AgentPlanProposal(
            goal_summary="Open the requested calculator.",
            success_conditions=("Calculator is open.",),
            rationale="The user requested one application to be opened.",
            uncertainty=0.1,
            steps=(
                {"action": "open", "target": "calculator"},
            ),
        )

    @staticmethod
    def task_metadata(proposal):
        return {
            "agent_mode": "cognitive_v1",
            "agent_goal_summary": proposal.goal_summary,
            "agent_success_conditions": list(proposal.success_conditions),
            "agent_rationale": proposal.rationale,
            "agent_uncertainty": proposal.uncertainty,
        }


def test_planner_accepts_cognitive_proposal_and_preserves_provenance():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    brain = FakeBrain()

    planner = Planner(
        kernel.tool_registry,
        agent_brain=brain,
    )

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="open calculator",
        entities={"action": "open", "target": "calculator"},
        requires_tools=True,
        classifier="rules",
    )

    plan = planner.plan(
        "open calculator",
        intent=intent,
    )

    assert brain.calls
    assert plan.metadata["planner"] == "cognitive_v1"
    assert plan.metadata["agent_goal_summary"] == "Open the requested calculator."
    assert plan.metadata["agent_success_conditions"] == ["Calculator is open."]
    assert plan.steps[0].description == "open calculator"
