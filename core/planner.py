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
        agent_brain=None,
    ):
        self.selector = ToolSelector(registry)
        self.media_manager = media_manager
        self.application_manager = application_manager
        self.agent_brain = agent_brain

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

        planner_name = "deterministic"
        agent_metadata: dict[str, Any] = {}
        brain = self.agent_brain
        if brain is not None and getattr(brain, "enabled", False):
            try:
                proposal = brain.plan(value, intent=intent)
                commands = [dict(step) for step in proposal.steps]
                planner_name = "cognitive_v1"
                agent_metadata = brain.task_metadata(proposal)
                print(
                    f"[Agent] Goal: {proposal.goal_summary}",
                    flush=True,
                )
                print(
                    f"[Agent] Success conditions: "
                    f"{'; '.join(proposal.success_conditions)}",
                    flush=True,
                )
                print(
                    f"[Agent] Rationale: {proposal.rationale}",
                    flush=True,
                )
                print(
                    f"[Agent] Uncertainty: {proposal.uncertainty:.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[Agent] Cognitive planning unavailable; "
                    f"falling back to deterministic planning: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        elif brain is not None:
            print("[Agent] Cognitive planning disabled (ASTA_AGENT_MODE=0).", flush=True)

        commands = self._expand_media_commands(commands)
        commands = self._normalize_grounded_computer_commands(commands)

        if planner_name == "cognitive_v1":
            commands = self._normalize_visual_verification_commands(
                commands,
                success_conditions=proposal.success_conditions,
                goal=value,
            )

        commands = self._deduplicate_adjacent_commands(commands)

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
                "planner": planner_name,
                "intent_type": intent.intent.value,
                "intent_confidence": intent.confidence,
                **agent_metadata,
                **dict(metadata or {}),
            },
        )
        print(
            "[Tasks] Plan: "
            + " -> ".join(step.description for step in steps),
            flush=True,
        )
        return plan

    def _normalize_grounded_computer_commands(
        self,
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert vision.locate before semantic coordinate-based computer actions."""
        normalized: list[dict[str, Any]] = []

        for command in commands:
            current = dict(command)
            action = str(current.get("action") or "").strip().lower()
            target = str(current.get("target") or "").strip()
            has_coordinates = (
                current.get("x") is not None
                and current.get("y") is not None
            )

            if action == "click" and target and not has_coordinates:
                previous = normalized[-1] if normalized else None
                previous_action = str(
                    (previous or {}).get("action") or ""
                ).strip().lower()
                previous_tool = str(
                    (previous or {}).get("tool") or ""
                ).strip()
                previous_target = str(
                    (previous or {}).get("target") or ""
                ).strip()

                already_grounded = (
                    previous_action == "locate"
                    and previous_tool == "vision.locate"
                    and previous_target.lower() == target.lower()
                )

                if not already_grounded:
                    try:
                        self.selector.select(
                            IntentResult(
                                intent=IntentType.COMMAND,
                                confidence=1.0,
                                normalized_text="",
                                entities={
                                    "action": "locate",
                                    "target": target,
                                },
                                requires_tools=True,
                                classifier="planner_grounding",
                            )
                        )
                    except ValueError:
                        pass
                    else:
                        normalized.append(
                            {
                                "action": "locate",
                                "tool": "vision.locate",
                                "target": target,
                            }
                        )

            normalized.append(current)

        return normalized

    def _normalize_visual_verification_commands(
        self,
        commands: list[dict[str, Any]],
        *,
        success_conditions: tuple[str, ...] | list[str],
        goal: str,
    ) -> list[dict[str, Any]]:
        """Make visual verification a deterministic semantic step."""
        conditions_text = " ".join(
            str(item).strip()
            for item in success_conditions
            if str(item).strip()
        )
        conditions_lower = conditions_text.lower()

        visual_required = any(
            marker in conditions_lower
            for marker in (
                "visible",
                "visually",
                "screenshot",
                "screen",
                "window",
                "display",
                "ui",
                "interface",
            )
        )
        if not visual_required:
            return commands

        normalized = [
            command
            for command in commands
            if str(command.get("tool") or "").strip()
            not in {"vision.screenshot", "vision.open_screenshot"}
        ]

        has_inspector = any(
            str(command.get("tool") or "").strip() == "vision.inspect"
            or str(command.get("action") or "").strip().lower()
            in {"inspect", "visual_verify"}
            for command in normalized
        )
        if has_inspector:
            return normalized

        prompt = (
            f"Verify the user's goal from the current screenshot. "
            f"Goal: {goal}. Success conditions: {conditions_text}. "
            "Set visual_match=true only when the screenshot supports "
            "the success conditions. Return concise evidence."
        )

        try:
            self.selector.select(
                IntentResult(
                    intent=IntentType.COMMAND,
                    confidence=1.0,
                    normalized_text=goal,
                    entities={
                        "action": "inspect",
                        "prompt": prompt,
                    },
                    requires_tools=True,
                    classifier="planner_visual_verification",
                )
            )
        except ValueError:
            return normalized

        normalized.append(
            {
                "action": "inspect",
                "tool": "vision.inspect",
                "prompt": prompt,
            }
        )
        return normalized

    @staticmethod
    def _deduplicate_adjacent_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove accidental consecutive duplicate cognitive actions."""
        deduplicated: list[dict[str, Any]] = []
        previous_signature = None
        for command in commands:
            signature = tuple(
                (key, str(command.get(key) or "").strip().lower())
                for key in (
                    "action",
                    "tool",
                    "target",
                    "operation",
                    "query",
                    "provider",
                )
                if command.get(key) is not None
            )
            if signature == previous_signature:
                continue
            deduplicated.append(dict(command))
            previous_signature = signature
        return deduplicated

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
        """Expand media intent according to whether the user chose a GUI flow."""
        expanded: list[dict[str, Any]] = []

        for index, command in enumerate(commands):
            normalized = dict(command)
            action = str(normalized.get("action") or "").strip().lower()
            if action != "media":
                expanded.append(normalized)
                continue

            operation = str(normalized.get("operation") or "").strip().lower()
            query = str(normalized.get("query") or "").strip()
            provider = str(normalized.get("provider") or "").strip()

            # An explicit preceding "open <app>" means the user asked A.S.T.A.
            # to operate the application through its visible UI. Keep that path
            # generic: use the registered computer/vision primitives instead of
            # silently switching to a provider API or capability setup flow.
            previous = commands[index - 1] if index > 0 else None
            previous_action = str(
                (previous or {}).get("action") or ""
            ).strip().lower()
            previous_target = str(
                (previous or {}).get("target") or ""
            ).strip()
            if (
                operation == "play"
                and query
                and previous_action in {"open", "launch", "start"}
                and previous_target
            ):
                expanded.extend(
                    self._interactive_media_play_steps(
                        query=query,
                        application=previous_target,
                    )
                )
                continue

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

    def _interactive_media_play_steps(
        self,
        *,
        query: str,
        application: str = "",
    ) -> list[dict[str, Any]]:
        """Build a provider-agnostic GUI search/play sequence."""
        application_text = application.strip() or "the target application"
        search_target = (
            f"the search input field used to enter a query in {application_text}"
        )
        result_target = (
            f"the visible song or track result titled '{query}' "
            f"in {application_text}"
        )
        verification_prompt = (
            f"Verify that '{query}' is actually playing in {application_text}. "
            "Inspect only the target application's own UI. Ignore A.S.T.A.'s HUD, "
            "conversation panel, assistant messages, subtitles, terminal output, "
            "or any overlay/text that merely repeats the requested item or command. "
            "Require concrete in-app playback evidence such as the requested track, "
            "artist/title, and an active playback indicator or player state. "
            "Set visual_match=true only when that evidence is visible in the supplied screenshot."
        )
        return [
            {
                "action": "locate",
                "tool": "vision.locate",
                "target": search_target,
            },
            {
                "action": "click",
                "tool": "computer.click",
                "target": search_target,
            },
            {
                "action": "type_text",
                "tool": "computer.type_text",
                "text": query,
            },
            {
                "action": "keypress",
                "tool": "computer.keypress",
                "key": "enter",
            },
            {
                "action": "locate",
                "tool": "vision.locate",
                "target": result_target,
            },
            {
                "action": "click",
                "tool": "computer.click",
                "target": result_target,
            },
            {
                "action": "inspect",
                "tool": "vision.inspect",
                "prompt": verification_prompt,
            },
        ]

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
