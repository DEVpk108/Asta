from abc import ABC, abstractmethod

from core.contracts import (
    ToolDefinition,
    ToolRequest,
    ToolResult,
)


class Tool(ABC):
    """
    Base contract for every executable ASTA tool.

    Tools own capability-specific logic.
    They do not perform intent classification,
    model selection, or global policy decisions.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the immutable tool definition."""
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        """
        Execute a validated tool request.

        Tool implementations should return a ToolResult
        rather than raising expected operational errors.
        """
        raise NotImplementedError