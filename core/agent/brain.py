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
class AgentDecision:
    """Structured post-action decision for the cognitive runtime loop."""

    goal_satisfied: bool = False
    needs_observation: bool = False
    needs_user: bool = False
    rationale: str = ""
    confidence: float = 0.5
    next_action: dict[str, Any] | None = None
    belief_updates: tuple[dict[str, Any], ...] = ()
    uncertainty: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_satisfied": self.goal_satisfied,
            "needs_observation": self.needs_observation,
            "needs_user": self.needs_user,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "next_action": dict(self.next_action) if self.next_action else None,
            "belief_updates": [dict(item) for item in self.belief_updates],
            "uncertainty": self.uncertainty,
        }


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

            # Agent planning/decision is structured control traffic. Keep its
            # output budget small and deterministic so the general conversation
            # model settings do not make every control step expensive.
            try:
                max_output = int(
                    os.getenv("ASTA_AGENT_MAX_OUTPUT_TOKENS", "128")
                )
                if hasattr(self._provider, "max_output_tokens"):
                    self._provider.max_output_tokens = max(64, max_output)
                if hasattr(self._provider, "reasoning_retry_tokens"):
                    self._provider.reasoning_retry_tokens = max(64, max_output)
            except (TypeError, ValueError):
                pass

            try:
                temperature = float(
                    os.getenv("ASTA_AGENT_TEMPERATURE", "0.0")
                )
                if hasattr(self._provider, "temperature"):
                    self._provider.temperature = max(0.0, temperature)
            except (TypeError, ValueError):
                pass
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

    def decide(self, task_state: dict[str, Any]) -> AgentDecision:
        """Interpret the latest observation and choose the next bounded action."""
        if not self.enabled:
            raise AgentBrainError("agent mode is disabled")

        prompt = self._build_decision_prompt(task_state)
        provider = self.provider()

        reset = getattr(provider, "reset_conversation", None)
        if callable(reset):
            reset()

        response = provider.generate_response(prompt, context=None)
        decision = self._parse_decision(response)
        decision = self._normalize_visual_next_action(
            decision,
            task_state,
        )
        decision = self._enforce_goal_verification_gate(
            decision,
            task_state,
        )
        self._validate_decision(decision)
        return decision

    def _build_decision_prompt(self, task_state: dict[str, Any]) -> str:
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

        payload = {
            "task": self._compact_task_state(task_state),
            "capabilities": self._compact_capabilities(registry),
            "instruction": (
                "You are the post-action decision layer of A.S.T.A. "
                "Compare the latest result with the goal and success conditions. "
                "Never claim visual or external verification without evidence. "
                "For visual verification, use vision.inspect. When an action needs a "
                "specific screen coordinate, use vision.locate first and then use the grounded "
                "computer control tool. Do not open a raw screenshot with vision.open_screenshot "
                "when semantic inspection is available. Choose the smallest useful next action. If user input is required, set needs_user=true. "
                "Return compact JSON only; no hidden chain-of-thought."
            ),
            "required_output": {
                "goal_satisfied": "boolean",
                "needs_observation": "boolean",
                "needs_user": "boolean",
                "rationale": "one concise sentence",
                "confidence": "number from 0 to 1",
                "uncertainty": "number from 0 to 1",
                "next_action": {
                    "action": "semantic action",
                    "tool": "exact registered tool name when needed",
                    "target": "optional target",
                    "prompt": "optional visual inspection prompt",
                    "operation": "optional operation",
                    "query": "optional query",
                    "provider": "optional provider"
                },
                "belief_updates": "array; keep empty unless a new belief is useful",
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _normalize_visual_next_action(
        self,
        decision: AgentDecision,
        task_state: dict[str, Any],
    ) -> AgentDecision:
        """Route visual screenshot-opening decisions to semantic inspection."""
        next_action = decision.next_action
        if not isinstance(next_action, dict):
            return decision

        if not self._goal_requires_visual_evidence(task_state):
            return decision

        tool = str(next_action.get("tool") or "").strip()
        action = str(next_action.get("action") or "").strip().lower()
        if tool != "vision.open_screenshot" and action != "open":
            return decision

        inspector = self._find_action_capability(task_state, action="inspect")
        if inspector is None:
            return decision

        updated = dict(next_action)
        updated.update(
            {
                "action": "inspect",
                "tool": inspector.name,
                "prompt": self._build_visual_verification_prompt(task_state),
            }
        )
        return AgentDecision(
            goal_satisfied=decision.goal_satisfied,
            needs_observation=True if not decision.goal_satisfied else decision.needs_observation,
            needs_user=decision.needs_user,
            rationale=decision.rationale,
            confidence=decision.confidence,
            next_action=updated,
            belief_updates=decision.belief_updates,
            uncertainty=decision.uncertainty,
        )

    def _enforce_goal_verification_gate(
        self,
        decision: AgentDecision,
        task_state: dict[str, Any],
    ) -> AgentDecision:
        """Prevent unsupported visual completion claims.

        Prefer the semantic vision inspector when available. A raw screenshot
        is only a fallback capability; it is not verification by itself.
        """
        if not decision.goal_satisfied:
            return decision

        if not self._goal_requires_visual_evidence(task_state):
            return decision

        if self._has_verified_visual_observation(task_state):
            return decision

        latest_tool = str(
            (task_state.get("latest_result") or {}).get("tool") or ""
        ).strip()

        inspector = self._find_action_capability(
            task_state,
            action="inspect",
        )
        if inspector is not None and latest_tool != inspector.name:
            return AgentDecision(
                goal_satisfied=False,
                needs_observation=True,
                needs_user=False,
                rationale=(
                    "The tool succeeded, but the goal requires visual evidence. "
                    "Use the semantic vision inspector before declaring success."
                ),
                confidence=min(decision.confidence, 0.85),
                next_action={
                    "action": "inspect",
                    "tool": inspector.name,
                    "prompt": self._build_visual_verification_prompt(task_state),
                },
                belief_updates=decision.belief_updates,
                uncertainty=max(decision.uncertainty, 0.25),
            )

        # A raw screenshot path is not itself a visual interpretation. Keep the
        # task open until a semantic vision result is available.
        screenshot = self._find_action_capability(
            task_state,
            action="screenshot",
        )
        inspector_name = inspector.name if inspector is not None else ""
        if latest_tool not in {inspector_name, "vision.screenshot"} and screenshot is not None:
            return AgentDecision(
                goal_satisfied=False,
                needs_observation=True,
                needs_user=False,
                rationale=(
                    "No semantic vision capability is available, so capture a "
                    "screenshot as evidence before declaring success."
                ),
                confidence=min(decision.confidence, 0.70),
                next_action={
                    "action": "screenshot",
                    "tool": screenshot.name,
                },
                belief_updates=decision.belief_updates,
                uncertainty=max(decision.uncertainty, 0.35),
            )

        return AgentDecision(
            goal_satisfied=False,
            needs_observation=True,
            needs_user=False,
            rationale=(
                "The available evidence does not contain a semantic visual "
                "verification of the goal."
            ),
            confidence=min(decision.confidence, 0.5),
            next_action=None,
            belief_updates=decision.belief_updates,
            uncertainty=max(decision.uncertainty, 0.5),
        )

    @staticmethod
    def _build_visual_verification_prompt(task_state: dict[str, Any]) -> str:
        goal = str(task_state.get("goal") or "").strip()
        conditions = task_state.get("success_conditions") or ()
        condition_text = "; ".join(
            str(item).strip()
            for item in conditions
            if str(item).strip()
        )
        if condition_text:
            return (
                f"Verify the user's goal from the current screenshot. "
                f"Goal: {goal}. Success conditions: {condition_text}. "
                "Set visual_match=true only when the screenshot itself supports "
                "the success conditions. Return concise evidence."
            )
        return (
            f"Verify the user's goal from the current screenshot. Goal: {goal}. "
            "Set visual_match=true only when the screenshot itself supports the goal. "
            "Return concise evidence."
        )

    @staticmethod
    def _goal_requires_visual_evidence(task_state: dict[str, Any]) -> bool:
        conditions = task_state.get("success_conditions") or ()
        text = " ".join(str(item) for item in conditions).lower()
        markers = (
            "visible",
            "visually",
            "screenshot",
            "screen",
            "window",
            "display",
            "ui",
            "interface",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _has_verified_visual_observation(task_state: dict[str, Any]) -> bool:
        observations = task_state.get("observations") or ()
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            data = observation.get("data")
            if not isinstance(data, dict):
                continue
            output = data.get("output")
            if not isinstance(output, dict):
                output = data

            analysis = output.get("analysis")
            if isinstance(analysis, dict):
                merged = dict(output)
                merged.update(analysis)
                output = merged

            explicit_flags = (
                "verified",
                "vision_verified",
                "visual_match",
                "window_detected",
            )
            if any(bool(output.get(key)) for key in explicit_flags):
                return True

            # A tool may return a structured visibility result directly.
            if "visible" in output and bool(output.get("visible")):
                return True

        return False

    def _find_action_capability(
        self,
        task_state: dict[str, Any],
        *,
        action: str,
    ):
        registry = getattr(self.kernel, "tool_registry", None)
        if registry is None:
            return None

        from core.tools.selector import ToolSelector

        intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=1.0,
            normalized_text=str(task_state.get("goal") or ""),
            entities={
                "action": action,
                # Capability discovery can inspect a definition before a
                # concrete request exists. The semantic vision inspector has
                # a required prompt, which is supplied after selection.
                **({"prompt": "(deferred)" } if action == "inspect" else {}),
            },
            requires_tools=True,
            classifier="agent_verification_gate",
        )
        try:
            return ToolSelector(registry).select(intent)
        except ValueError:
            return None

    @staticmethod
    def _compact_capabilities(registry) -> list[dict[str, Any]]:
        """Serialize only control-relevant tool metadata for the brain prompt."""
        if registry is None:
            return []

        compact = []
        for definition in registry.definitions():
            metadata = definition.metadata if isinstance(definition.metadata, dict) else {}
            schema = definition.input_schema if isinstance(definition.input_schema, dict) else {}
            compact.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "actions": list(metadata.get("actions") or metadata.get("action_aliases") or ()),
                    "category": metadata.get("category"),
                    "required_inputs": list(schema.get("required") or ()),
                }
            )
        return compact

    @staticmethod
    def _compact_task_state(task_state: dict[str, Any]) -> dict[str, Any]:
        """Bound agent decision prompts to the state needed for the next action."""
        payload = {
            "goal": task_state.get("goal"),
            "success_conditions": list(task_state.get("success_conditions") or ()),
            "beliefs": dict(task_state.get("beliefs") or {}),
            "current_strategy": task_state.get("current_strategy", ""),
            "uncertainty": task_state.get("uncertainty", 1.0),
            "current_step": task_state.get("current_step"),
            "latest_result": task_state.get("latest_result"),
        }

        observations = list(task_state.get("observations") or ())
        if observations:
            payload["latest_observation"] = observations[-1]

        # The full action/decision history is useful for journaling but creates
        # large repeated prompts. Keep only the latest decision for control.
        if task_state.get("last_decision") is not None:
            payload["last_decision"] = task_state.get("last_decision")

        return payload

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = str(response or "").strip()
        if not text:
            raise AgentBrainError("agent decision returned an empty response")

        candidates = [text]
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

        if not isinstance(parsed, dict):
            raise AgentBrainError("agent decision returned invalid JSON")

        raw_action = parsed.get("next_action")
        next_action = dict(raw_action) if isinstance(raw_action, dict) else None

        raw_updates = parsed.get("belief_updates") or []
        if not isinstance(raw_updates, list):
            raw_updates = []
        belief_updates = tuple(
            dict(item)
            for item in raw_updates
            if isinstance(item, dict)
        )

        try:
            raw_confidence = parsed.get("confidence", 0.5)
            raw_uncertainty = parsed.get("uncertainty", 0.5)
            confidence = float(
                0.5 if raw_confidence is None else raw_confidence
            )
            uncertainty = float(
                0.5 if raw_uncertainty is None else raw_uncertainty
            )
        except (TypeError, ValueError) as exc:
            raise AgentBrainError("agent decision confidence is invalid") from exc

        return AgentDecision(
            goal_satisfied=bool(parsed.get("goal_satisfied", False)),
            needs_observation=bool(parsed.get("needs_observation", False)),
            needs_user=bool(parsed.get("needs_user", False)),
            rationale=str(parsed.get("rationale") or "").strip(),
            confidence=max(0.0, min(1.0, confidence)),
            next_action=next_action,
            belief_updates=belief_updates,
            uncertainty=max(0.0, min(1.0, uncertainty)),
        )

    def _validate_decision(self, decision: AgentDecision) -> None:
        if not decision.rationale:
            raise AgentBrainError("agent decision omitted rationale")

        if decision.goal_satisfied and decision.next_action:
            raise AgentBrainError(
                "agent decision cannot be satisfied and request another action"
            )

        if decision.needs_user and decision.next_action:
            raise AgentBrainError(
                "agent decision cannot require user input and request an action"
            )

        if decision.next_action is None:
            if not (
                decision.goal_satisfied
                or decision.needs_observation
                or decision.needs_user
            ):
                raise AgentBrainError(
                    "agent decision returned neither completion nor a next action"
                )
            return

        registry = getattr(self.kernel, "tool_registry", None)
        if registry is None:
            raise AgentBrainError("tool registry is unavailable")

        action = str(
            decision.next_action.get("action") or ""
        ).strip().lower()
        if not action:
            raise AgentBrainError("agent decision next_action omitted action")

        normalized = dict(decision.next_action)
        if registry.contains(action):
            definition = registry.get(action).definition
            metadata = (
                definition.metadata
                if isinstance(definition.metadata, dict)
                else {}
            )
            semantic_actions = metadata.get("actions")
            if not isinstance(
                semantic_actions,
                (list, tuple, set, frozenset),
            ):
                semantic_actions = metadata.get("action_aliases")
            if isinstance(
                semantic_actions,
                (list, tuple, set, frozenset),
            ):
                semantic_actions = [
                    str(item).strip().lower()
                    for item in semantic_actions
                    if str(item).strip()
                ]
            else:
                semantic_actions = []
            if semantic_actions:
                normalized["tool"] = action
                normalized["action"] = semantic_actions[0]
                action = semantic_actions[0]

        entities = {
            key: value
            for key, value in normalized.items()
            if value is not None and value != ""
        }
        entities["action"] = action
        intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=decision.confidence,
            normalized_text="",
            entities=entities,
            requires_tools=True,
            classifier="agent_brain_decision",
        )
        from core.tools.selector import ToolSelector
        try:
            definition = ToolSelector(registry).select(intent)
        except ValueError as exc:
            raise AgentBrainError(
                f"agent decision next_action is not executable: {exc}"
            ) from exc
        normalized["tool"] = definition.name
        decision.next_action.clear()
        decision.next_action.update(normalized)

    def _build_prompt(self, goal: str, intent: IntentResult) -> str:
        registry = getattr(self.kernel, "tool_registry", None)
        capabilities = self._plan_capabilities(registry, goal, intent)

        payload = {
            "goal": goal,
            "intent": intent.entities,
            "capabilities": capabilities,
            "instruction": (
                "Create the smallest safe executable plan. Return JSON only. "
                "For goals that require visual verification, return the primary "
                "action only; A.S.T.A. will append the semantic vision inspection "
                "step deterministically. When a computer action needs an exact "
                "screen coordinate, do not invent coordinates: plan vision.locate "
                "first and let the next decision use its grounded result. Do not "
                "add screenshots or open-screenshot steps. Keep success conditions "
                "observable and concise."
            ),
            "required_output": {
                "goal_summary": "one sentence",
                "success_conditions": ["1-3 observable conditions"],
                "rationale": "one concise sentence",
                "uncertainty": "number from 0 to 1",
                "steps": [
                    {
                        "action": "semantic action",
                        "tool": "exact registered tool name",
                        "target": "optional target",
                        "operation": "optional operation",
                        "query": "optional query",
                        "provider": "optional provider"
                    }
                ]
            }
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _plan_capabilities(registry, goal: str, intent: IntentResult) -> list[dict[str, Any]]:
        if registry is None:
            return []

        from core.tools.selector import ToolSelector

        selected = []
        try:
            definition = ToolSelector(registry).select(intent)
            selected.append(definition)
        except (KeyError, ValueError):
            pass

        goal_text = str(goal or "").lower()
        visual_goal = any(
            marker in goal_text
            for marker in (
                "verify",
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
        selected_category = ""
        if selected:
            selected_definition = selected[0]
            selected_category = str(
                (selected_definition.metadata or {}).get("category") or ""
            ).strip().lower()

        if (
            (visual_goal or selected_category == "computer")
            and registry.contains("vision.locate")
        ):
            selected.append(registry.get("vision.locate").definition)

        if visual_goal and registry.contains("vision.inspect"):
            selected.append(registry.get("vision.inspect").definition)

        if not selected:
            return AgentBrain._compact_capabilities(registry)

        deduped = []
        seen = set()
        for definition in selected:
            if definition.name in seen:
                continue
            seen.add(definition.name)
            metadata = definition.metadata if isinstance(definition.metadata, dict) else {}
            schema = definition.input_schema if isinstance(definition.input_schema, dict) else {}
            deduped.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "actions": list(metadata.get("actions") or metadata.get("action_aliases") or ()),
                    "category": metadata.get("category"),
                    "required_inputs": list(schema.get("required") or ()),
                }
            )
        return deduped

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
            parsed = AgentBrain._repair_truncated_plan(text)

        if not isinstance(parsed, dict):
            raise AgentBrainError("agent planner returned invalid JSON")

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
            raw_uncertainty = parsed.get("uncertainty", 0.5)
            uncertainty = float(
                0.5 if raw_uncertainty is None else raw_uncertainty
            )
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

    @staticmethod
    def _repair_truncated_plan(text: str) -> dict[str, Any] | None:
        """Recover completed step objects from a truncated JSON response."""
        if '"steps"' not in text:
            return None

        def scalar(pattern: str, default: str = ""):
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else default

        goal_summary = scalar(r'"goal_summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', "")
        rationale = scalar(r'"rationale"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', "")
        uncertainty_match = re.search(r'"uncertainty"\s*:\s*([0-9.]+)', text)
        try:
            uncertainty = float(uncertainty_match.group(1)) if uncertainty_match else 0.5
        except ValueError:
            uncertainty = 0.5

        conditions = []
        conditions_match = re.search(r'"success_conditions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if conditions_match:
            conditions = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', conditions_match.group(1))

        array_start = text.find("[", text.find('"steps"'))
        if array_start < 0:
            return None

        objects = []
        i = array_start + 1
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n or text[i] != "{":
                break

            obj_start = i
            depth = 0
            in_string = False
            escape = False
            completed = False
            while i < n:
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                objects.append(json.loads(text[obj_start:i + 1]))
                            except json.JSONDecodeError:
                                pass
                            i += 1
                            completed = True
                            break
                i += 1
            if not completed:
                break
            while i < n and text[i] not in "{]":
                i += 1

        if not objects:
            return None

        return {
            "goal_summary": goal_summary,
            "success_conditions": conditions,
            "rationale": rationale,
            "uncertainty": uncertainty,
            "steps": objects,
        }

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

            normalized = dict(step)
            # Models may copy an exact capability name into action even
            # though the executor expects the capability semantic action.
            if registry.contains(action):
                definition = registry.get(action).definition
                metadata = definition.metadata if isinstance(definition.metadata, dict) else {}
                semantic_actions = metadata.get("actions")
                if not isinstance(semantic_actions, (list, tuple, set, frozenset)):
                    semantic_actions = metadata.get("action_aliases")
                if isinstance(semantic_actions, (list, tuple, set, frozenset)):
                    semantic_actions = [
                        str(item).strip().lower()
                        for item in semantic_actions
                        if str(item).strip()
                    ]
                else:
                    semantic_actions = []
                if semantic_actions:
                    normalized["tool"] = action
                    normalized["action"] = semantic_actions[0]
                    action = semantic_actions[0]

            if action == "click":
                target = str(normalized.get("target") or "").strip()
                has_coordinates = (
                    normalized.get("x") is not None
                    and normalized.get("y") is not None
                )
                if not target and not has_coordinates:
                    raise AgentBrainError(
                        f"agent planner step {index} click action must include a target or coordinates"
                    )
                if target and not has_coordinates:
                    normalized.pop("x", None)
                    normalized.pop("y", None)

            entities = {
                key: value
                for key, value in normalized.items()
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

            # Keep the normalized representation for the deterministic planner.
            step.clear()
            step.update(normalized)

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
