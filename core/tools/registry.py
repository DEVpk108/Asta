from core.tools.base import Tool


class ToolRegistry:
    """
    Registry of capabilities available to ASTA.

    The registry is responsible only for tool discovery.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.definition.name.strip()

        if not name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        if name in self._tools:
            raise ValueError(
                f"Tool already registered: {name}"
            )

        self._tools[name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(
            name,
            None,
        ) is not None

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown tool: {name}"
            ) from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> tuple[str, ...]:
        return tuple(self._tools.keys())

    def definitions(self) -> tuple:
        return tuple(
            tool.definition
            for tool in self._tools.values()
        )