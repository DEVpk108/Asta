from __future__ import annotations

import time

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


def _result(request: ToolRequest, success: bool, *, output=None, error=None, start: float) -> ToolResult:
    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


class CreateNoteTool(Tool):
    def __init__(self, notes_manager):
        self.notes_manager = notes_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notes.create_note",
            description="Create a persistent local Markdown note.",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                    "title": {"type": "string"},
                },
                "required": ["content"],
            },
            risk_level="low",
            metadata={
                "actions": ["create_note"],
                "action_aliases": ["write_note", "take_note", "save_note"],
                "category": "notes",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        content = request.arguments.get("content")
        title = request.arguments.get("title")
        if not isinstance(content, str) or not content.strip():
            return _result(request, False, error="Argument 'content' must be a non-empty string.", start=start)
        if title is not None and not isinstance(title, str):
            return _result(request, False, error="Argument 'title' must be a string.", start=start)

        try:
            output = self.notes_manager.create(content, title=title)
        except FileExistsError as exc:
            return _result(request, False, error=str(exc), start=start)
        except Exception as exc:
            return _result(request, False, error=f"Failed to create note: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output=output, start=start)


class ReadNoteTool(Tool):
    def __init__(self, notes_manager):
        self.notes_manager = notes_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notes.read_note",
            description="Read a persistent local note by title.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string", "minLength": 1}},
                "required": ["target"],
            },
            risk_level="low",
            metadata={
                "actions": ["read_note"],
                "action_aliases": ["open_note", "show_note"],
                "category": "notes",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        target = request.arguments.get("target")
        if not isinstance(target, str) or not target.strip():
            return _result(request, False, error="Argument 'target' must be a non-empty string.", start=start)

        try:
            output = self.notes_manager.read(target)
        except FileNotFoundError as exc:
            return _result(request, False, error=str(exc), start=start)
        except Exception as exc:
            return _result(request, False, error=f"Failed to read note: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output=output, start=start)


class ListNotesTool(Tool):
    def __init__(self, notes_manager):
        self.notes_manager = notes_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notes.list_notes",
            description="List persistent local notes.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["list_notes"],
                "action_aliases": ["show_notes"],
                "category": "notes",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        try:
            notes = self.notes_manager.list()
        except Exception as exc:
            return _result(request, False, error=f"Failed to list notes: {type(exc).__name__}: {exc}", start=start)
        return _result(request, True, output={"count": len(notes), "notes": notes}, start=start)


class SearchNotesTool(Tool):
    def __init__(self, notes_manager):
        self.notes_manager = notes_manager

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="notes.search_notes",
            description="Search persistent local notes by text.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
            risk_level="low",
            metadata={
                "actions": ["search_notes"],
                "action_aliases": ["find_notes"],
                "category": "notes",
            },
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        query = request.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return _result(request, False, error="Argument 'query' must be a non-empty string.", start=start)

        try:
            matches = self.notes_manager.search(query)
        except Exception as exc:
            return _result(request, False, error=f"Failed to search notes: {type(exc).__name__}: {exc}", start=start)
        return _result(
            request,
            True,
            output={"query": query.strip(), "count": len(matches), "matches": matches},
            start=start,
        )


__all__ = ["CreateNoteTool", "ReadNoteTool", "ListNotesTool", "SearchNotesTool"]
