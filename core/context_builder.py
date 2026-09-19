from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import IntentResult, IntentType


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Runtime context assembled for one reasoning turn."""

    user_text: str
    task: dict[str, Any] | None = None
    memory: str = ""
    workspace: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[dict[str, Any], ...] = ()
    skills: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "task": dict(self.task) if self.task else None,
            "memory": self.memory,
            "workspace": dict(self.workspace),
            "capabilities": [dict(item) for item in self.capabilities],
            "skills": [dict(item) for item in self.skills],
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

        if self.skills:
            skill_lines = ["RELEVANT SKILLS FOR THIS TURN:"]
            for skill in self.skills:
                skill_lines.append(
                    f"- {skill['name']}: {skill['description']}\n"
                    f"  Instructions: {skill['instructions']}"
                )
            sections.append("\n".join(skill_lines))

        if self.capabilities:
            capability_lines = ["AVAILABLE CAPABILITIES FOR THIS TURN:"]
            for item in self.capabilities:
                capability_lines.append(
                    f"- {item['name']}: {item['description']} "
                    f"[provider={item.get('provider', 'native')}, "
                    f"risk={item.get('risk_level', 'unknown')}, "
                    f"confirmation={item.get('requires_confirmation', False)}, "
                    f"loaded={item.get('loaded', True)}]"
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
    discovers relevant skills/capabilities, and produces a structured snapshot
    that can later grow to include MCP and other context providers.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.discovery = getattr(kernel, "capability_discovery", None)
        self.skill_manager = getattr(kernel, "skill_manager", None)

    def build(self, user_text: str, intent: IntentResult | None = None) -> ContextSnapshot:
        normalized = str(user_text).strip()
        task = self._task_snapshot()
        memory = str(getattr(self.kernel, "memory_context", "") or "").strip()
        workspace = self._workspace_snapshot()
        skills = self._skills_for(intent, normalized)
        capabilities = self._capabilities_for(intent)

        return ContextSnapshot(
            user_text=normalized,
            task=task,
            memory=memory,
            workspace=workspace,
            capabilities=capabilities,
            skills=skills,
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

        value = getattr(self.kernel, "workspace_context", None)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _skills_for(
        self,
        intent: IntentResult | None,
        user_text: str,
    ) -> tuple[dict[str, Any], ...]:
        if self.skill_manager is None:
            return ()

        try:
            skills = self.skill_manager.discover(
                intent,
                query=user_text,
                limit=3,
            )
        except Exception:
            return ()

        return tuple(
            {
                "name": skill.name,
                "description": skill.description,
                "instructions": skill.instructions,
                "provider": skill.provider,
                "priority": skill.priority,
            }
            for skill in skills
        )

    def _capabilities_for(
        self,
        intent: IntentResult | None,
    ) -> tuple[dict[str, Any], ...]:
        if intent is None or intent.intent is not IntentType.COMMAND:
            return ()

        if self.discovery is not None:
            intents = self._command_intents(intent)
            descriptors = self.discovery.discover_commands(
                intents,
                limit_per_intent=3,
            )
            return tuple(descriptor.to_dict() for descriptor in descriptors)

        return ()

    @staticmethod
    def _command_intents(intent: IntentResult) -> list[IntentResult]:
        commands = intent.entities.get("commands")
        if not isinstance(commands, list) or not commands:
            return [intent]

        results: list[IntentResult] = []
        for command in commands:
            if not isinstance(command, dict):
                continue
            results.append(
                IntentResult(
                    intent=IntentType.COMMAND,
                    confidence=intent.confidence,
                    normalized_text=intent.normalized_text,
                    entities=dict(command),
                    requires_tools=True,
                    classifier=intent.classifier,
                )
            )
        return results
