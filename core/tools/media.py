from __future__ import annotations

import time

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.media import MediaRequest
from core.tools.base import Tool


def _result(
    request: ToolRequest,
    success: bool,
    *,
    output=None,
    error=None,
    start: float,
) -> ToolResult:
    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


class MediaControlTool(Tool):
    def __init__(self, media_manager):
        self.media_manager = media_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="media.control",
            description=(
                "Control local media playback. Supports play, pause, toggle, "
                "next, previous, and stop, with an optional media query and provider."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "play",
                            "pause",
                            "toggle",
                            "next",
                            "previous",
                            "stop",
                        ],
                    },
                    "query": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "provider": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["media"],
                "category": "media",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()

        operation = request.arguments.get("operation")
        query = request.arguments.get("query")
        provider = request.arguments.get("provider")

        if not isinstance(operation, str) or not operation.strip():
            return _result(
                request,
                False,
                error="Argument 'operation' must be a non-empty string.",
                start=start,
            )

        if query is not None and (
            not isinstance(query, str) or not query.strip()
        ):
            return _result(
                request,
                False,
                error="Argument 'query' must be a non-empty string when provided.",
                start=start,
            )

        if provider is not None and (
            not isinstance(provider, str) or not provider.strip()
        ):
            return _result(
                request,
                False,
                error="Argument 'provider' must be a non-empty string when provided.",
                start=start,
            )

        result = self.media_manager.execute(
            MediaRequest(
                operation=operation.strip().lower(),
                query=query.strip() if isinstance(query, str) else None,
                provider=provider.strip().lower() if isinstance(provider, str) else None,
            )
        )

        output = dict(result.output or {})
        output.update(
            {
                "provider": result.provider,
                "operation": result.operation,
                "query": result.query,
                "message": result.message,
            }
        )
        return _result(
            request,
            result.success,
            output=output,
            error=result.error,
            start=start,
        )
