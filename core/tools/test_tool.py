import time

from core.contracts import (
    ToolDefinition,
    ToolRequest,
    ToolResult,
)

from core.tools.base import Tool


class EchoTool(Tool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description=(
                "Returns the supplied text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    },
                },
                "required": [
                    "text",
                ],
            },
            risk_level="low",
            requires_confirmation=False,
            timeout_seconds=5.0,
        )

    def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:

        start = time.perf_counter()

        text = request.arguments.get(
            "text"
        )

        if not isinstance(text, str):
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=(
                    "Argument 'text' "
                    "must be a string."
                ),
            )

        return ToolResult(
            success=True,
            tool=self.definition.name,
            output=text,
            duration_seconds=(
                time.perf_counter()
                - start
            ),
        )