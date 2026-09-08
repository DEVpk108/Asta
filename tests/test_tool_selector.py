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
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            metadata={"actions": ["open"]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(success=True, tool=self.definition.name, output=request.arguments)


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
    kernel.register_tool(FakeOpenTool())
    selected = ToolSelector(kernel.tool_registry).select(command_intent())
    assert selected.name == "test.open_application"


def test_request_builder_preserves_selected_action_metadata():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    request = ToolRequestBuilder(kernel.tool_registry).build(command_intent())
    assert request.tool == "test.open_application"
    assert request.arguments == {"target": "calculator"}
    assert request.metadata["action"] == "open"


def test_selector_rejects_missing_required_entity():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    intent = command_intent(target=None)
    intent = IntentResult(
        intent=intent.intent,
        confidence=intent.confidence,
        normalized_text="open",
        entities={"action": "open"},
        requires_tools=True,
        classifier="rules",
    )
    try:
        ToolSelector(kernel.tool_registry).select(intent)
    except ValueError as exc:
        assert "No registered tool" in str(exc)
    else:
        raise AssertionError("Selector should reject a tool with missing required input")
