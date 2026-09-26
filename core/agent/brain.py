from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from core.contracts import IntentResult, IntentType


class AgentBrainError(RuntimeError):
    """Raised when the cognitive planner cannot produce a valid decision."""


@dataclass(frozen=True, slots=True)
class AgentPlanProposal:
    goal_summary: str
    success_conditions: tuple[str, ...]
    rationale: str
    steps: tuple[dict[str, Any], ...]
    uncertainty: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_summary": self.goal_summary,
            "success_conditions": list(self.success_conditions),
            "rationale": self.rationale,
            "steps": [dict(step) for step in self.steps],
            "uncertainty": self.uncertainty,
        }


class AgentBrain:
    """LLM-backed cognitive planner above the deterministic executor.

    V1 proposes structured intent steps. Existing Planner/TaskRuntime remain
    authoritative for tool validation, sequencing and execution.
    """

    def __init__(
        self,
        kernel,
        *,
        provider=None,
        enabled: bool | None = None,
        provider_factory: Callable[[], Any] | None = None,
    ):
        self.kernel = kernel
        self._provider = provider
        self._provider_factory = provider_factory
        self._enabled_override = enabled
        self._system_prompt = (
            "You are the cognitive planning layer of A.S.T.A., a local-first "
            "AI engineering agent. Understand the user's goal, inspect the "
            "supplied state, identify observable success conditions, and "
            "propose a minimal executable plan using only supplied capabilities. "
            "Do not claim actions already happened. Return JSON only. Keep the "
            "rationale concise and provide a decision summary rather than hidden "
            "chain-of-thought. Include uncertainty when important state is unknown."
        )

    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return os.getenv("ASTA_AGENT_MODE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def provider(self):
        if self._provider is None:
            if self._provider_factory is not None:
                self._provider = self._provider_factory()
            else:
                from ai.llm_provider import create_llm_provider

                self._provider = create_llm_provider()

            setter = getattr(self._provider, "set_system_prompt", None)
            if callable(setter):
                setter(self._system_prompt)
        return self._provider

    def plan(self, goal: str, *, intent: IntentResult) -> AgentPlanProposal:
        if not self.enabled:
            raise AgentBrainError("agent mode is disabled")

        prompt = self._build_prompt(goal, intent)
        provider = self.provider()

        reset = getattr(provider, "reset_conversation", None)
        if callable(reset):
            reset()

        response = provider.generate_response(prompt, context=None)

        proposal = self._parse_response(response)
        self._validate_proposal(proposal)
        return proposal

    def _build_prompt(self, goal: str, intent: IntentResult) -> str:
        definitions = []
        registry = getattr(self.kernel, "tool_registry", None)
        if registry is not None:
            for definition in registry.definitions():
                definitions.append(
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "input_schema": definition.input_schema,
                        "risk_level": definition.risk_level,
                        "requires_confirmation": definition.requires_confirmation,
                        "metadata": definition.metadata,
                    }
                )

        workspace = {}
        workspace_manager = getattr(self.kernel, "workspace_manager", None)
        if workspace_manager is not None:
            try:
                workspace = workspace_manager.snapshot()
            except Exception:
                workspace = {}

        current_task = None
        task_manager = getattr(self.kernel, "task_manager", None)
        if task_manager is not None:
            try:
                current_task = task_manager.snapshot()
            except Exception:
                current_task = None

        payload = {
            "goal": goal,
            "intent": intent.entities,
            "intent_confidence": intent.confidence,
            "current_task": current_task,
            "workspace": workspace,
            "capabilities": definitions,
            "required_output": {
                "goal_summary": "one sentence",
                "success_conditions": ["1-3 observable conditions"],
                "rationale": "one or two concise sentences",
                "uncertainty": "number from 0 to 1",
                "steps": [
                    {
                        "action": "supported tool action such as open or media",
                        "target": "optional target",
                        "operation": "optional operation",
                        "query": "optional query",
                        "provider": "optional provider",
                    }
                ],
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _parse_response(response: str) -> AgentPlanProposal:
        text = str(response or "").strip()
        if not text:
            raise AgentBrainError("agent planner returned an empty response")

        candidates = [text]
        # Models sometimes wrap JSON in prose. Extract the outermost object as
        # a bounded fallback instead of depending on a particular markdown style.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])

        parsed = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

        if parsed is None:
            raise AgentBrainError("agent planner returned invalid JSON")

        if not isinstance(parsed, dict):
            raise AgentBrainError("agent planner root must be a JSON object")

        steps = parsed.get("steps")
        if not isinstance(steps, list) or not steps:
            raise AgentBrainError("agent planner returned no executable steps")

        clean_steps: list[dict[str, Any]] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise AgentBrainError(
                    f"agent planner step {index} is not an object"
                )
            clean_steps.append(dict(step))

        conditions = parsed.get("success_conditions") or []
        if not isinstance(conditions, list):
            conditions = [str(conditions)]

        try:
            uncertainty = float(parsed.get("uncertainty", 0.5) or 0.5)
        except (TypeError, ValueError) as exc:
            raise AgentBrainError("agent planner uncertainty is invalid") from exc

        return AgentPlanProposal(
            goal_summary=str(parsed.get("goal_summary") or "").strip(),
            success_conditions=tuple(
                str(item).strip()
                for item in conditions
                if str(item).strip()
            ),
            rationale=str(parsed.get("rationale") or "").strip(),
            steps=tuple(clean_steps),
            uncertainty=max(0.0, min(1.0, uncertainty)),
        )

    def _validate_proposal(self, proposal: AgentPlanProposal) -> None:
        if not proposal.goal_summary:
            raise AgentBrainError("agent planner omitted goal_summary")
        if not proposal.success_conditions:
            raise AgentBrainError("agent planner omitted success_conditions")
        if not proposal.rationale:
            raise AgentBrainError("agent planner omitted rationale")

        registry = getattr(self.kernel, "tool_registry", None)
        if registry is None:
            raise AgentBrainError("tool registry is unavailable")

        from core.tools.selector import ToolSelector

        selector = ToolSelector(registry)

        for index, step in enumerate(proposal.steps, start=1):
            action = str(step.get("action") or "").strip().lower()
            if not action:
                raise AgentBrainError(
                    f"agent planner step {index} omitted action"
                )

            entities = {
                key: value
                for key, value in step.items()
                if value is not None and value != ""
            }
            entities["action"] = action
            intent = IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.5,
                normalized_text=proposal.goal_summary,
                entities=entities,
                requires_tools=True,
                classifier="agent_brain",
            )
            try:
                selector.select(intent)
            except ValueError as exc:
                raise AgentBrainError(
                    f"agent planner step {index} is not executable: {exc}"
                ) from exc

    @staticmethod
    def task_metadata(proposal: AgentPlanProposal) -> dict[str, Any]:
        return {
            "agent_mode": "cognitive_v1",
            "agent_goal_summary": proposal.goal_summary,
            "agent_success_conditions": list(proposal.success_conditions),
            "agent_rationale": proposal.rationale,
            "agent_uncertainty": proposal.uncertainty,
            "agent_steps": [dict(step) for step in proposal.steps],
        }
