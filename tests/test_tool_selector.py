from core import Kernel
from core.contracts import IntentResult, IntentType, ToolDefinition, ToolRequest, ToolResult
from core.tools import Tool, ToolRequestBuilder, ToolSelector


class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open_application",
            description="Test capability for opening an application.",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
                "required": ["target"],
            },
            metadata={"actions": ["open"]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output=request.arguments,
        )


def command_intent(action="open", target="calculator"):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.95,
        normalized_text=f"{action} {target}",
        entities={"action": action, "target": target},
        requires_tools=True,
        classifier="rules",
    )


def test_selector_discovers_tool_from_registry_metadata():
    kernel = Kernel()
    tool = FakeOpenTool()
    kernel.register_tool(tool)

    selector = ToolSelector(kernel.tool_registry)
    selected = selector.select(command_intent())

    assert selected.name == "test.open_application"


def test_request_builder_uses_selected_definition_and_timeout():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())

    builder = ToolRequestBuilder(kernel.tool_registry)
    request = builder.build(command_intent())

    assert request.tool == "test.open_application"
    assert request.arguments == {"target": "calculator"}
    assert request.metadata["intent"] == "command"
    assert request.metadata["classifier"] == "rules"
