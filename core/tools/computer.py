from __future__ import annotations

import time

from core.computer import ComputerControlError, ComputerController
from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool


def _result(request, success, *, output=None, error=None, start):
    return ToolResult(
        success=success,
        tool=request.tool,
        output=output,
        error=error,
        duration_seconds=time.perf_counter() - start,
        metadata={"request_id": request.request_id},
    )


class _ComputerTool(Tool):
    def __init__(self, controller=None):
        self.controller = controller or ComputerController()


class ComputerMoveMouseTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.move_mouse",
            description="Move the mouse pointer to a screen coordinate.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "duration": {"type": "number", "minimum": 0},
                },
                "required": [],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["move_mouse"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        try:
            output = self.controller.move_mouse(
                x=request.arguments.get("x"),
                y=request.arguments.get("y"),
                duration=request.arguments.get("duration", 0.0),
            )
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Mouse move failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        return _result(request, True, output=output, start=start)


class ComputerClickTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.click",
            description="Click a screen location with the mouse.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "target": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional semantic target; task runtime may ground it from vision.locate.",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "middle", "right"],
                    },
                    "clicks": {"type": "integer", "minimum": 1},
                    "interval": {"type": "number", "minimum": 0},
                },
                "required": [],
                "additionalProperties": False,
            },
            risk_level="medium",
            metadata={
                "actions": ["click"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        x = request.arguments.get("x")
        y = request.arguments.get("y")
        if (x is None) != (y is None):
            return _result(
                request,
                False,
                error="Grounded click requires both x and y coordinates.",
                start=start,
            )
        if x is None or y is None:
            return _result(
                request,
                False,
                error="Grounded click coordinates were not supplied.",
                start=start,
            )
        try:
            output = self.controller.click(
                x=x,
                y=y,
                button=request.arguments.get("button", "left"),
                clicks=request.arguments.get("clicks", 1),
                interval=request.arguments.get("interval", 0.0),
            )
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Mouse click failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        if "target" in request.arguments:
            output["target"] = request.arguments.get("target")
        return _result(request, True, output=output, start=start)


class ComputerTypeTextTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.type_text",
            description="Type text into the currently focused application.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "interval": {"type": "number", "minimum": 0},
                    "paste": {"type": "boolean"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk_level="medium",
            metadata={
                "actions": ["type_text", "type"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        try:
            output = self.controller.type_text(
                request.arguments.get("text"),
                interval=request.arguments.get("interval", 0.0),
                paste=bool(request.arguments.get("paste", True)),
            )
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Text entry failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        return _result(request, True, output=output, start=start)


class ComputerKeypressTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.keypress",
            description="Press one keyboard key in the focused application.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "minLength": 1},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
            risk_level="medium",
            metadata={
                "actions": ["keypress", "press_key"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        try:
            output = self.controller.keypress(request.arguments.get("key"))
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Keypress failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        return _result(request, True, output=output, start=start)


class ComputerHotkeyTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.hotkey",
            description="Press a multi-key keyboard shortcut.",
            input_schema={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["keys"],
                "additionalProperties": False,
            },
            risk_level="medium",
            metadata={
                "actions": ["hotkey", "key_combo"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        try:
            output = self.controller.hotkey(request.arguments.get("keys"))
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Hotkey failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        return _result(request, True, output=output, start=start)


class ComputerScrollTool(_ComputerTool):
    @property
    def definition(self):
        return ToolDefinition(
            name="computer.scroll",
            description="Scroll the focused application or page.",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "integer"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["amount"],
                "additionalProperties": False,
            },
            risk_level="low",
            metadata={
                "actions": ["scroll"],
                "category": "computer",
            },
        )

    def execute(self, request):
        start = time.perf_counter()
        try:
            output = self.controller.scroll(
                request.arguments.get("amount"),
                x=request.arguments.get("x"),
                y=request.arguments.get("y"),
            )
        except Exception as exc:
            return _result(
                request,
                False,
                error=f"Scroll failed: {type(exc).__name__}: {exc}",
                start=start,
            )
        return _result(request, True, output=output, start=start)


__all__ = [
    "ComputerController",
    "ComputerMoveMouseTool",
    "ComputerClickTool",
    "ComputerTypeTextTool",
    "ComputerKeypressTool",
    "ComputerHotkeyTool",
    "ComputerScrollTool",
]
