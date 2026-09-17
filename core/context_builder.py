from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import IntentResult, IntentType
from .tools.selector import ToolSelector


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Runtime context assembled for one reasoning turn."""

    user_text: str
    task: dict[str, Any] | None = None
    memory: str = ""
    workspace: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "task": dict(self.task) if self.task else None,
            "memory": self.memory,
            "workspace": dict(self.workspace),
            "capabilities": [dict(item) for item in self.capabilities],
        }

    def to_prompt(self) -> str:
        sections = ["A.S.T.A. RUNTIME CONTEXT"]

        if self.task:
            task = self.task
            sections.append(
                "\n".join(
                    [
                        "ACTIVE TASK:",
                        f"Goal: {task.get('goal', '')}",
                        f"Status: {task.get('status', '')}",
                        f"Current step: {task.get('current_step') or 'none'}",
                        self._format_list("Completed steps", task.get("completed_steps")),
                        self._format_list("Pending steps", task.get("pending_steps")),
                        self._format_list("Constraints", task.get("constraints")),
                    ]
                )
            )

        if self.memory.strip():
            sections.append(f"LONG-TERM MEMORY:\n{self.memory.strip()}")

        if self.workspace:
            sections.append(self._format_mapping("WORKSPACE STATE", self.workspace))

        if self.capabilities:
            capability_lines = ["AVAILABLE CAPABILITIES FOR THIS TURN:"]
            capability_lines.extend(
                f"- {item['name']}: {item['description']}"
                for item in self.capabilities
            )
            sections.append("\n".join(capability_lines))

        sections.append(f"USER REQUEST:\n{self.user_text.strip()}")
        return "\n\n".join(sections)

    @staticmethod
    def _format_list(label: str, values: Any) -> str:
        if not values:
            return f"{label}: none"
        return f"{label}: " + "; ".join(str(value) for value in values)

    @staticmethod
    def _format_mapping(label: str, values: dict[str, Any]) -> str:
        lines = [f"{label}:"]
        lines.extend(f"- {key}: {value}" for key, value in values.items())
        return "\n".join(lines)


class ContextBuilder:
    """Assemble the smallest useful runtime context for one model turn.

    The builder is deliberately LLM-agnostic. It reads state from the kernel,
    selects only command-relevant capabilities, and produces a structured
    snapshot that can later grow to include skills, MCP, and other context
    providers.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.selector = ToolSelector(kernel.tool_registry)

    def build(self, user_text: str, intent: IntentResult | None = None) -> ContextSnapshot:
        normalized = str(user_text).strip()
        task = self._task_snapshot()
        memory = str(getattr(self.kernel, "memory_context", "") or "").strip()
        workspace = self._workspace_snapshot()
        capabilities = self._capabilities_for(intent)

        return ContextSnapshot(
            user_text=normalized,
            task=task,
            memory=memory,
            workspace=workspace,
            capabilities=capabilities,
        )

    def build_prompt(self, user_text: str, intent: IntentResult | None = None) -> str:
        return self.build(user_text, intent).to_prompt()

    def _task_snapshot(self) -> dict[str, Any] | None:
        manager = getattr(self.kernel, "task_manager", None)
        if manager is None:
            return None

        try:
            return manager.snapshot()
        except (AttributeError, TypeError):
            task = manager.current()
            return task.to_dict() if task is not None else None

    def _workspace_snapshot(self) -> dict[str, Any]:
        manager = getattr(self.kernel, "workspace_manager", None)
        if manager is not None:
            try:
                return manager.snapshot()
            except (AttributeError, TypeError):
                pass

        # Keep the original hook as a compatibility fallback for callers that
        # populated workspace_context before WorkspaceManager existed.
        value = getattr(self.kernel, "workspace_context", None)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _capabilities_for(
        self,
        intent: IntentResult | None,
    ) -> tuple[dict[str, Any], ...]:
        if intent is None or intent.intent is not IntentType.COMMAND:
            return ()

        definitions = []
        commands = intent.entities.get("commands")

        if isinstance(commands, list) and commands:
            for command in commands:
                if not isinstance(command, dict):
                    continue
                step_intent = IntentResult(
                    intent=IntentType.COMMAND,
                    confidence=intent.confidence,
                    normalized_text=intent.normalized_text,
                    entities=dict(command),
                    requires_tools=True,
                    classifier=intent.classifier,
                )
                self._append_selected(definitions, step_intent)
        else:
            self._append_selected(definitions, intent)

        return tuple(definitions)

    def _append_selected(
        self,
        definitions: list[dict[str, Any]],
        intent: IntentResult,
    ) -> None:
        try:
            definition = self.selector.select(intent)
        except ValueError:
            return

        item = {
            "name": definition.name,
            "description": definition.description,
        }
        if item not in definitions:
            definitions.append(item)
