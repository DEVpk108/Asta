from core import Kernel, Planner
from core.contracts import IntentResult, IntentType, PlanStatus, ToolDefinition
from core.tools import Tool


class FakeTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open something for tests.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request):
        raise AssertionError("planner tests must not execute tools")


class AnotherFakeTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.close",
            description="Close something for tests.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["close"]},
        )

    def execute(self, request):
        raise AssertionError("planner tests must not execute tools")


def _planner():
    kernel = Kernel()
    kernel.register_tool(FakeTool())
    kernel.register_tool(AnotherFakeTool())
    return Planner(kernel.tool_registry)


def _command_intent(**entities):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="test command",
        entities=entities,
        requires_tools=True,
        classifier="rules",
    )


def test_planner_creates_ready_plan_for_single_command():
    planner = _planner()

    plan = planner.plan(
        "open calculator",
        intent=_command_intent(action="open", target="calculator"),
    )

    assert plan.status is PlanStatus.READY
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "open calculator"
    assert plan.steps[0].required_capabilities == ["test.open"]
    assert plan.steps[0].completion_conditions == [
        "test.open reports success"
    ]


def test_planner_chains_compound_commands():
    planner = _planner()

    plan = planner.plan(
        "open calculator and then close calculator",
        intent=_command_intent(
            commands=[
                {"action": "open", "target": "calculator"},
                {"action": "close", "target": "calculator"},
            ]
        ),
    )

    assert [step.id for step in plan.steps] == ["step-1", "step-2"]
    assert plan.steps[0].depends_on == []
    assert plan.steps[1].depends_on == ["step-1"]
    assert [step.metadata["tool"] for step in plan.steps] == [
        "test.open",
        "test.close",
    ]


def test_planner_does_not_execute_tools():
    planner = _planner()

    plan = planner.plan(
        "open calculator",
        intent=_command_intent(action="open", target="calculator"),
    )

    assert plan.steps[0].metadata["tool"] == "test.open"


def test_planner_rejects_non_command_intents():
    planner = _planner()

    intent = IntentResult(
        intent=IntentType.UNKNOWN,
        confidence=0.20,
        normalized_text="diagnose my project",
    )

    try:
        planner.plan("diagnose my project", intent=intent)
    except ValueError as exc:
        assert "only supports command intents" in str(exc)
    else:
        raise AssertionError("Expected planner rejection")
