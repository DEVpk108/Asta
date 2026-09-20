from __future__ import annotations

from typing import Any

from core.contracts import (
    IntentResult,
    IntentType,
    Plan,
    PlanStatus,
    PlanStep,
)
from core.tools.selector import ToolSelector


class PlanningError(ValueError):
    """Raised when a goal cannot be converted into a safe structured plan."""


class Planner:
    """Create validated execution plans from structured intents.

    This first planner is deliberately deterministic. It converts command
    intents into sequential PlanSteps and records the selected capability for
    each step. Natural-language engineering goals that are not yet represented
    as command intents are rejected here; a later LLM-backed strategy can
    implement that planning surface without changing the Plan contract.
    """

    def __init__(self, registry):
        self.selector = ToolSelector(registry)

    def plan(
        self,
        goal: str,
        *,
        intent: IntentResult,
        constraints: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Plan:
        value = str(goal).strip()
        if not value:
            raise PlanningError("plan goal must be non-empty")

        if not isinstance(intent, IntentResult):
            raise PlanningError("planner requires an IntentResult")

        if intent.intent is not IntentType.COMMAND:
            raise PlanningError(
                "deterministic planner only supports command intents"
            )

        commands = self._commands_from_intent(intent)
        if not commands:
            raise PlanningError("command intent contains no executable commands")

        steps: list[PlanStep] = []
        previous_id: str | None = None

        for index, command in enumerate(commands, start=1):
            command_intent = IntentResult(
                intent=IntentType.COMMAND,
                confidence=intent.confidence,
                normalized_text=intent.normalized_text,
                entities=dict(command),
                requires_tools=True,
                classifier=intent.classifier,
            )

            try:
                definition = self.selector.select(command_intent)
            except ValueError as exc:
                raise PlanningError(str(exc)) from exc

            action = str(command.get("action") or "").strip()
            target = str(command.get("target") or "").strip()
            description = " ".join(
                part for part in (action, target) if part
            )

            step_id = f"step-{index}"
            steps.append(
                PlanStep(
                    id=step_id,
                    description=description or definition.name,
                    depends_on=[previous_id] if previous_id else [],
                    required_capabilities=[definition.name],
                    completion_conditions=[
                        f"{definition.name} reports success",
                    ],
                    metadata={
                        "action": action,
                        "target": target,
                        "tool": definition.name,
                        "sequence_index": index - 1,
                    },
                )
            )
            previous_id = step_id

        plan = Plan(
            goal=value,
            steps=steps,
            constraints=constraints or (),
            status=PlanStatus.READY,
            metadata={
                "planner": "deterministic",
                "intent_type": intent.intent.value,
                "intent_confidence": intent.confidence,
                **dict(metadata or {}),
            },
        )
        return plan

    @staticmethod
    def _commands_from_intent(intent: IntentResult) -> list[dict[str, Any]]:
        commands = intent.entities.get("commands")
        if isinstance(commands, list):
            return [
                dict(command)
                for command in commands
                if isinstance(command, dict)
            ]

        return [dict(intent.entities)]
