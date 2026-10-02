from core.agent.brain import AgentBrain
from core.contracts import IntentResult, IntentType, ToolDefinition
from core.tools.registry import ToolRegistry


class FakeKernel:
    def __init__(self):
        self.tool_registry = ToolRegistry()


def register(registry, name, category, actions, required):
    from core.tools.base import Tool

    class DefinitionTool(Tool):
        @property
        def definition(self):
            return ToolDefinition(
                name=name,
                description=name,
                input_schema={
                    "type": "object",
                    "properties": {
                        key: {} for key in required
                    },
                    "required": list(required),
                },
                metadata={
                    "actions": list(actions),
                    "category": category,
                },
            )

        def execute(self, request):
            raise NotImplementedError

    registry.register(DefinitionTool())


def test_agent_planner_exposes_locate_for_coordinate_computer_actions():
    kernel = FakeKernel()
    register(
        kernel.tool_registry,
        "computer.click",
        "computer",
        ["click"],
        ["x", "y"],
    )
    register(
        kernel.tool_registry,
        "vision.locate",
        "vision",
        ["locate"],
        ["target"],
    )

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="click create app",
        entities={
            "action": "click",
            "x": 700,
            "y": 400,
        },
        requires_tools=True,
        classifier="rules",
    )

    capabilities = AgentBrain._plan_capabilities(
        kernel.tool_registry,
        "click the Create App button",
        intent,
    )
    names = {item["name"] for item in capabilities}

    assert "computer.click" in names
    assert "vision.locate" in names


def test_agent_planner_exposes_locate_when_visual_goal_has_no_direct_tool():
    kernel = FakeKernel()
    register(
        kernel.tool_registry,
        "vision.locate",
        "vision",
        ["locate"],
        ["target"],
    )
    register(
        kernel.tool_registry,
        "vision.inspect",
        "vision",
        ["inspect"],
        ["prompt"],
    )

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.90,
        normalized_text="verify what is on screen",
        entities={
            "action": "inspect",
            "prompt": "verify the target",
        },
        requires_tools=True,
        classifier="rules",
    )

    capabilities = AgentBrain._plan_capabilities(
        kernel.tool_registry,
        "verify the target on screen",
        intent,
    )
    names = {item["name"] for item in capabilities}

    assert "vision.locate" in names
    assert "vision.inspect" in names



def test_planner_inserts_locate_before_targeted_click():
    from core.contracts import IntentResult, IntentType, ToolDefinition
    from core.planner import Planner
    from core.tools.base import Tool
    from core.tools.registry import ToolRegistry

    class DefinitionTool(Tool):
        def __init__(self, name, action, properties, required):
            self._definition = ToolDefinition(
                name=name,
                description=name,
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                metadata={"actions": [action], "category": "test"},
            )

        @property
        def definition(self):
            return self._definition

        def execute(self, request):
            raise NotImplementedError

    registry = ToolRegistry()
    registry.register(
        DefinitionTool(
            "vision.locate",
            "locate",
            {"target": {"type": "string"}},
            ["target"],
        )
    )
    registry.register(
        DefinitionTool(
            "computer.click",
            "click",
            {
                "target": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            [],
        )
    )

    planner = Planner(registry)

    commands = planner._normalize_grounded_computer_commands(
        [
            {
                "action": "click",
                "tool": "computer.click",
                "target": "Create App",
            }
        ]
    )

    assert commands == [
        {
            "action": "locate",
            "tool": "vision.locate",
            "target": "Create App",
        },
        {
            "action": "click",
            "tool": "computer.click",
            "target": "Create App",
        },
    ]
