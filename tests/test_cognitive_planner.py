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


def test_planner_accepts_cognitive_proposal_and_preserves_provenance(monkeypatch):
    # These cover the LLM planner itself; force it for a rules command.
    monkeypatch.setenv("ASTA_AGENT_PLANNING", "always")
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


class FakeWriteTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.write",
            description="Write a file for tests.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["write_file"]},
        )

    def execute(self, request):
        raise AssertionError("planner tests must not execute tools")


class FakeProjectBrain(FakeBrain):
    def plan(self, goal, *, intent):
        self.calls.append((goal, intent.entities))
        return AgentPlanProposal(
            goal_summary="Write hello.py.",
            success_conditions=("hello.py exists.",),
            rationale="The user asked for one file.",
            uncertainty=0.1,
            steps=({"action": "write_file", "path": "hello.py", "content": "print(1)\n"},),
        )


def test_task_runs_explicit_app_commands_before_llm_project_steps():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    kernel.register_tool(FakeWriteTool())
    brain = FakeProjectBrain()
    planner = Planner(kernel.tool_registry, agent_brain=brain)
    request = "open spotify, then write hello.py"
    intent = IntentResult(
        intent=IntentType.TASK,
        confidence=0.9,
        normalized_text=request,
        entities={
            "goal": "write hello.py",
            "request": request,
            "capability_scope": ["test.write"],
            "prefix_commands": [{"action": "open", "target": "spotify"}],
        },
        requires_tools=True,
        requires_reasoning=True,
        classifier="project_task_rules",
    )

    plan = planner.plan(request, intent=intent)

    goal, entities = brain.calls[0]
    assert goal == "write hello.py"
    assert "prefix_commands" not in entities
    assert entities["capability_scope"] == ["test.write"]
    assert [step.metadata["tool"] for step in plan.steps] == ["test.open", "test.write"]
    assert plan.steps[0].metadata["target"] == "spotify"
    assert plan.goal == request


def test_mixed_request_plays_through_the_music_apps_reliable_path(monkeypatch):
    from core.tools import UIPlayTool
    from core.tools.computer import ComputerWaitTool

    monkeypatch.delenv("ASTA_SPOTIFY_CLIENT_ID", raising=False)
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    kernel.register_tool(FakeWriteTool())
    kernel.register_tool(ComputerWaitTool())
    kernel.register_tool(UIPlayTool(runner=lambda *a: ""))
    brain = FakeProjectBrain()
    planner = Planner(
        kernel.tool_registry,
        agent_brain=brain,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )
    request = "open spotify and play baithi hai, then write hello.py"
    intent = IntentResult(
        intent=IntentType.TASK,
        confidence=0.9,
        normalized_text=request,
        entities={
            "goal": "write hello.py",
            "request": request,
            "capability_scope": ["test.write"],
            "prefix_commands": [
                {"action": "open", "target": "spotify"},
                {"action": "media", "operation": "play", "query": "baithi hai"},
            ],
        },
        requires_tools=True,
        requires_reasoning=True,
        classifier="project_task_rules",
    )

    plan = planner.plan(request, intent=intent)

    tools = [step.metadata["tool"] for step in plan.steps]
    # Search link -> wait -> press the result's Play button (confirmed from the
    # media session), not the on-screen search-box hunt.
    assert tools == ["test.open", "computer.wait", "media.ui_play", "test.write"]
    assert plan.steps[0].metadata["target"].startswith("spotify:search:")
    assert plan.steps[2].metadata["process"] == "Spotify"
    assert all(step.metadata.get("user_request_prefix") is True for step in plan.steps[:3])
    assert "user_request_prefix" not in plan.steps[3].metadata
    assert "vision.inspect" not in tools
