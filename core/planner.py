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

    def __init__(
        self,
        registry,
        *,
        media_manager=None,
        application_manager=None,
    ):
        self.selector = ToolSelector(registry)
        self.media_manager = media_manager
        self.application_manager = application_manager

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

        commands = self._expand_media_commands(commands)

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

            description_parts = []
            if action == "media":
                operation = str(command.get("operation") or "media").strip()
                query = str(command.get("query") or "").strip()
                description_parts.extend(
                    part for part in (operation, query) if part
                )
            else:
                description_parts.extend(
                    part for part in (action, target) if part
                )
            description = " ".join(description_parts)

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
                        **(
                            {"verification": "media.playback"}
                            if (
                                action == "media"
                                and str(command.get("operation") or "").strip().lower() == "play"
                                and str(command.get("query") or "").strip()
                                and str(command.get("provider") or "").strip().lower() == "spotify"
                            )
                            else {}
                        ),
                        **{
                            key: value
                            for key, value in command.items()
                            if key not in {"action", "target"}
                            and value is not None
                            and value != ""
                        },
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
        print(
            "[Tasks] Plan: "
            + " -> ".join(step.description for step in steps),
            flush=True,
        )
        return plan

    def replan(self, task, diagnosis, strategy) -> Plan:
        """Build a bounded repair plan from the existing task state."""
        strategy_value = getattr(strategy, "value", str(strategy)).strip().lower()
        if strategy_value == "rebuild_plan":
            entities = dict(task.metadata.get("intent_entities") or {})
            intent = IntentResult(
                intent=IntentType.COMMAND,
                confidence=float(task.metadata.get("confidence", 0.98)),
                normalized_text=task.goal,
                entities=entities,
                requires_tools=True,
                classifier=str(task.metadata.get("classifier", "replan")),
            )
            return self.plan(
                task.goal,
                intent=intent,
                constraints=task.constraints,
                metadata={
                    "planner": "recovery_rebuild",
                    "replan_from": getattr(diagnosis, "step_id", None),
                    "diagnosis_category": getattr(
                        getattr(diagnosis, "category", None),
                        "value",
                        "unknown",
                    ),
                },
            )

        if strategy_value != "restore_state":
            raise PlanningError(
                f"Unsupported replan strategy '{strategy_value}'."
            )

        step_id = getattr(diagnosis, "step_id", None)
        if not step_id or task.plan is None:
            raise PlanningError("restore_state requires a failed plan step")

        failed = task.plan.get_step(step_id)
        action = str(failed.metadata.get("action") or "").strip().lower()
        target = str(failed.metadata.get("target") or "").strip()
        provider = str(failed.metadata.get("provider") or "").strip()

        restore_target = ""
        if provider and self.media_manager is not None:
            resolver = getattr(
                self.media_manager,
                "application_for_provider",
                None,
            )
            if callable(resolver):
                restore_target = str(resolver(provider) or "").strip()

        if not restore_target and target and action not in {"open", "launch", "start"}:
            restore_target = target

        if not restore_target:
            raise PlanningError(
                "No safe restore target is available for the failed step."
            )

        restore_intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=float(task.metadata.get("confidence", 0.98)),
            normalized_text=task.goal,
            entities={"action": "open", "target": restore_target},
            requires_tools=True,
            classifier="replan_restore_state",
        )
        definition = self.selector.select(restore_intent)
        restore_step = PlanStep(
            id="replan-restore",
            description=f"open {restore_target}",
            required_capabilities=[definition.name],
            completion_conditions=[f"{definition.name} reports success"],
            metadata={
                "action": "open",
                "target": restore_target,
                "tool": definition.name,
                "sequence_index": 0,
            },
        )

        retry_step = PlanStep(
            id="replan-retry",
            description=failed.description,
            depends_on=[restore_step.id],
            required_capabilities=list(failed.required_capabilities),
            completion_conditions=list(failed.completion_conditions),
            metadata={**dict(failed.metadata), "replanned": True},
        )

        return Plan(
            goal=task.goal,
            steps=[restore_step, retry_step],
            constraints=task.constraints,
            status=PlanStatus.READY,
            metadata={
                "planner": "recovery_restore_state",
                "replan_from": step_id,
                "diagnosis_category": getattr(
                    getattr(diagnosis, "category", None),
                    "value",
                    "unknown",
                ),
            },
        )

    def _expand_media_commands(
        self,
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expand provider-backed media play into setup + playback steps."""
        expanded: list[dict[str, Any]] = []

        for command in commands:
            normalized = dict(command)
            if str(normalized.get("action") or "").strip().lower() != "media":
                expanded.append(normalized)
                continue

            operation = str(normalized.get("operation") or "").strip().lower()
            query = str(normalized.get("query") or "").strip()
            provider = str(normalized.get("provider") or "").strip()

            if not provider and self.application_manager is not None:
                recent = getattr(
                    self.application_manager,
                    "last_opened_application",
                    None,
                )
                recent_name = getattr(recent, "name", recent)
                infer_provider = getattr(
                    self.media_manager,
                    "provider_for_application",
                    None,
                )
                if recent_name and callable(infer_provider):
                    provider = str(
                        infer_provider(str(recent_name)) or ""
                    ).strip()

            application = None
            if provider and self.media_manager is not None:
                resolve_app = getattr(
                    self.media_manager,
                    "application_for_provider",
                    None,
                )
                if callable(resolve_app):
                    application = resolve_app(provider)

            if operation == "play" and query and application:
                expanded.append(
                    {
                        "action": "open",
                        "target": str(application),
                    }
                )

            if provider:
                normalized["provider"] = provider

            expanded.append(normalized)

        return expanded

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
