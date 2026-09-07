from core.contracts import IntentResult, IntentType, ToolDefinition
from core.tools import Tool, ToolRegistry, ToolRequestBuilder, ToolSelector


class DummyTool(Tool):
    def __init__(self, definition):
        self._definition = definition

    @property
    def definition(self):
        return self._definition

    def execute(self, request):
        raise NotImplementedError


def command(action, **entities):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.99,
        normalized_text=f"{action} ...",
        entities={"action": action, **entities},
        requires_tools=True,
        classifier="rules",
    )


def test_selector_uses_registered_capability_metadata():
    registry = ToolRegistry()
    registry.register(
        DummyTool(
            ToolDefinition(
                name="system.open_application",
                description="Open an application.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                metadata={"actions": ["open"]},
            )
        )
    )

    selector = ToolSelector(registry)
    selected = selector.select(command("open", target="calculator"))

    assert selected.name == "system.open_application"


def test_selector_prefers_exact_action_over_alias():
    registry = ToolRegistry()

    registry.register(
        DummyTool(
            ToolDefinition(
                name="generic.launcher",
                description="Launch things.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                metadata={
                    "actions": ["launch"],
                    "action_aliases": ["start"],
                },
            )
        )
    )
    registry.register(
        DummyTool(
            ToolDefinition(
                name="process.start",
                description="Start a process.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                metadata={"actions": ["start"]},
            )
        )
    )

    selected = ToolSelector(registry).select(
        command("start", target="server")
    )

    assert selected.name == "process.start"


def test_selector_rejects_missing_required_entity():
    registry = ToolRegistry()
    registry.register(
        DummyTool(
            ToolDefinition(
                name="system.open_application",
                description="Open an application.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                metadata={"actions": ["open"]},
            )
        )
    )

    try:
        ToolSelector(registry).select(command("open"))
    except ValueError as exc:
        assert "No registered tool supports" in str(exc)
    else:
        raise AssertionError("Selection should fail without required inputs")


def test_request_builder_uses_selected_definition():
    registry = ToolRegistry()
    registry.register(
        DummyTool(
            ToolDefinition(
                name="system.open_application",
                description="Open an application.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                timeout_seconds=7.5,
                metadata={"actions": ["open"]},
            )
        )
    )

    request = ToolRequestBuilder(registry).build(
        command("open", target="calculator")
    )

    assert request.tool == "system.open_application"
    assert request.arguments == {"target": "calculator"}
    assert request.timeout_seconds == 7.5
    assert request.metadata["intent"] == "command"
