from __future__ import annotations

import os
import time
from typing import Any

from core.autonomy.diagnosis import DiagnosisCategory, FailureDiagnosis
from core.autonomy.replanning import ReplanDecision, ReplanStrategy
from core.contracts.action import ActionDecision, ActionType
from core.media import parse_media_request

from .base import DecisionEngine
from .contracts import DecisionSnapshot
from .laya_schemas import ASTA_DECISION_QUESTIONS
from .laya_action_schemas import build_action_questions
from .laya_diagnosis_schemas import build_diagnosis_questions
from .laya_replan_schemas import build_replan_questions


class LayaDecisionEngine(DecisionEngine):
    """Laya-backed System-1 decision provider.

    Laya is deliberately used as a decision/routing layer, not as A.S.T.A.'s
    main generative model and not as an authorization mechanism.
    """

    name = "laya"
    _SUPPORTED_MODELS = {
        "english",
        "multilingual",
    }

    def __init__(
        self,
        *,
        device: str | None = None,
        preload: bool | None = None,
        max_loaded: int | None = None,
        model: str | None = None,
    ):
        self.device = self._normalize_device(
            device or os.getenv("ASTA_LAYA_DEVICE")
        )
        self.model = self._normalize_model(
            model or os.getenv("ASTA_LAYA_MODEL", "multilingual")
        )
        self.preload = (
            self._env_bool("ASTA_LAYA_PRELOAD", True)
            if preload is None
            else bool(preload)
        )
        self.max_loaded = max(
            1,
            int(
                max_loaded
                if max_loaded is not None
                else os.getenv("ASTA_LAYA_MAX_LOADED", "1")
            ),
        )
        self._router = None
        self.warmed = False

    def _get_router(self):
        if self._router is not None:
            return self._router

        try:
            from laya import Router
        except ImportError as exc:
            raise RuntimeError(
                "Laya decision engine is enabled but the 'laya' package is not installed. "
                "Install requirements-laya.txt or disable ASTA_DECISION_ENGINE."
            ) from exc

        kwargs: dict[str, Any] = {
            "default": self.model,
            "max_loaded": self.max_loaded,
            "preload": False,
        }
        if self.device:
            kwargs["device"] = self.device

        self._router = Router(**kwargs)
        return self._router

    def warmup(self) -> bool:
        """Load the selected Laya checkpoint during A.S.T.A. startup."""
        try:
            router = self._get_router()
            if self.preload:
                router.preload([self.model])
            else:
                router.load(self.model)
            self.warmed = True
            print(
                f"[AI] Laya {self.model} warm-up complete.",
                flush=True,
            )
            return True
        except Exception as exc:
            self.warmed = False
            print(
                f"[AI] Laya warm-up unavailable: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False


    def decide_action(
        self,
        text: str,
        *,
        applications=None,
        media_providers=None,
    ) -> ActionDecision:
        """Select a computer action from a finite structured action space."""
        value = str(text or "").strip()
        if not value:
            return ActionDecision(source=self.name, model=self.model)

        names = []
        for application in applications or ():
            name = getattr(application, "name", application)
            name = str(name or "").strip()
            if name and name not in names:
                names.append(name)

        names = names[:128]
        providers = []
        for provider in media_providers or ():
            name = getattr(provider, "name", provider)
            name = str(name or "").strip()
            if name and name not in providers:
                providers.append(name)
        providers = providers[:32]

        questions = build_action_questions(names, providers)

        started = time.perf_counter()
        result = self._get_router().predict(
            {"user_request": value, "applications": names},
            questions,
            model=self.model,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answers = result.get("answers") or {}
        routing = result.get("routing") or {}
        model = routing.get("model") or result.get("model") or self.model

        def choice(question_id: str, default=None):
            answer = answers.get(question_id) or {}
            value = answer.get("choice")
            return value if value is not None else default

        def probability(question_id: str) -> float:
            answer = answers.get(question_id) or {}
            try:
                return float(answer.get("noul", 0.0))
            except (TypeError, ValueError):
                return 0.0

        action_value = str(choice("action", "none") or "none").strip().lower()
        try:
            action = ActionType(action_value)
        except ValueError:
            action = ActionType.NONE

        target_app = choice("target_app")
        arguments = {}
        if target_app and target_app != "none":
            arguments["target_app"] = str(target_app)

        media_operation = choice("media_operation")
        media_request = parse_media_request(value)
        if action is ActionType.MEDIA:
            if media_operation and media_operation != "none":
                arguments["operation"] = str(media_operation)
            if media_request is not None:
                if media_request.query:
                    arguments["query"] = media_request.query
                if media_request.provider:
                    arguments["provider"] = media_request.provider
                if "operation" not in arguments:
                    arguments["operation"] = media_request.operation

            media_provider = choice("media_provider")
            if (
                media_provider
                and media_provider != "none"
                and "provider" not in arguments
            ):
                arguments["provider"] = str(media_provider)

        addressed = probability("addressed")
        complete = probability("command_complete") >= 0.50
        compound = probability("compound") >= 0.50

        action_answer = answers.get("action") or {}
        try:
            confidence = float(action_answer.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if target_app and target_app != "none":
            target_answer = answers.get("target_app") or {}
            try:
                confidence = min(
                    confidence,
                    float(target_answer.get("confidence", confidence)),
                )
            except (TypeError, ValueError):
                pass

        if action is ActionType.MEDIA:
            operation_answer = answers.get("media_operation") or {}
            try:
                confidence = min(
                    confidence,
                    float(operation_answer.get("confidence", confidence)),
                )
            except (TypeError, ValueError):
                pass

        return ActionDecision(
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            arguments=arguments,
            addressed=max(0.0, min(1.0, addressed)),
            command_complete=complete,
            compound=compound,
            source=self.name,
            model=str(model) if model is not None else None,
            latency_ms=elapsed_ms,
            raw_decisions={
                str(key): dict(answer)
                for key, answer in answers.items()
                if isinstance(answer, dict)
            },
        )

    def diagnose_failure(self, state: dict[str, Any]) -> FailureDiagnosis:
        """Classify one failure using one bounded Laya forward pass."""
        questions = build_diagnosis_questions()
        started = time.perf_counter()
        result = self._get_router().predict(
            state,
            questions,
            model=self.model,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answers = result.get("answers") or {}
        routing = result.get("routing") or {}
        model = routing.get("model") or result.get("model") or self.model

        def choice(question_id: str, default: str) -> str:
            answer = answers.get(question_id) or {}
            value = answer.get("choice")
            return str(value or default).strip().lower()

        def probability(question_id: str) -> float:
            answer = answers.get(question_id) or {}
            try:
                return max(0.0, min(1.0, float(answer.get("noul", 0.0))))
            except (TypeError, ValueError):
                return 0.0

        category_value = choice("category", DiagnosisCategory.UNKNOWN.value)
        try:
            category = DiagnosisCategory(category_value)
        except ValueError:
            category = DiagnosisCategory.UNKNOWN

        recommended = choice("recommended_action", "replan")
        allowed_actions = {"retry", "wait_for_user", "replan", "fail"}
        if recommended not in allowed_actions:
            recommended = "replan"

        category_answer = answers.get("category") or {}
        action_answer = answers.get("recommended_action") or {}
        try:
            category_confidence = float(category_answer.get("confidence", 0.0))
        except (TypeError, ValueError):
            category_confidence = 0.0
        try:
            action_confidence = float(action_answer.get("confidence", category_confidence))
        except (TypeError, ValueError):
            action_confidence = category_confidence

        confidence = max(0.0, min(1.0, min(category_confidence, action_confidence)))
        requires_user = probability("requires_user") >= 0.50
        if requires_user:
            recommended = "wait_for_user"

        tool = str(state.get("failed_tool") or "")
        step_id = state.get("step_id")
        task_id = state.get("task_id")
        error = str(state.get("error") or "").strip()
        summary = (
            f"{tool or 'Tool execution'} appears to have failed because of "
            f"{category.value.replace('_', ' ')}."
        )

        return FailureDiagnosis(
            category=category,
            summary=summary,
            failed_tool=tool,
            task_id=str(task_id) if task_id else None,
            step_id=str(step_id) if step_id else None,
            attempt=int(state.get("attempt") or 1),
            confidence=confidence,
            recommended_action=recommended,
            requires_user=requires_user,
            source=self.name,
            model=str(model) if model is not None else None,
            latency_ms=elapsed_ms,
            evidence={
                "error": error[:1200],
                "output": str(state.get("output") or "")[:1200],
            },
            metadata={
                "routing": dict(routing),
                "raw_decisions": {
                    str(key): dict(answer)
                    for key, answer in answers.items()
                    if isinstance(answer, dict)
                },
            },
        )

    def select_replan_strategy(
        self,
        diagnosis,
        *,
        plan_step=None,
    ) -> ReplanDecision:
        """Choose one bounded repair strategy with a single Laya pass."""
        strategy_questions = build_replan_questions()
        step_payload = {}
        if plan_step is not None:
            step_payload = {
                "id": getattr(plan_step, "id", None),
                "description": getattr(plan_step, "description", ""),
                "status": getattr(getattr(plan_step, "status", None), "value", None),
                "metadata": dict(getattr(plan_step, "metadata", {}) or {}),
            }

        diagnosis_payload = (
            diagnosis.to_dict()
            if hasattr(diagnosis, "to_dict")
            else dict(diagnosis or {})
        )
        state = {
            "diagnosis": diagnosis_payload,
            "failed_step": step_payload,
        }

        started = time.perf_counter()
        result = self._get_router().predict(
            state,
            strategy_questions,
            model=self.model,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answers = result.get("answers") or {}
        routing = result.get("routing") or {}
        model = routing.get("model") or result.get("model") or self.model
        answer = answers.get("strategy") or {}
        raw_strategy = str(answer.get("choice") or "rebuild_plan").strip().lower()
        try:
            strategy = ReplanStrategy(raw_strategy)
        except ValueError:
            strategy = ReplanStrategy.REBUILD_PLAN

        try:
            confidence = float(answer.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # The user boundary never belongs to Laya. It is enforced by the
        # ReplanEngine before this provider is consulted.
        return ReplanDecision(
            strategy=strategy,
            reason=f"Laya selected {strategy.value} for the diagnosed failure.",
            confidence=max(0.0, min(1.0, confidence)),
            source=self.name,
            model=str(model) if model is not None else None,
            latency_ms=elapsed_ms,
            metadata={
                "routing": dict(routing),
                "raw_decision": dict(answer),
            },
        )

    def shutdown(self) -> None:
        router = self._router
        self._router = None
        self.warmed = False

        if router is None:
            return

        try:
            router.unload()
            print("[AI] Laya checkpoint unloaded.", flush=True)
        except Exception as exc:
            print(
                f"[AI] Laya shutdown error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def analyze(self, text: str) -> DecisionSnapshot:
        value = str(text or "").strip()
        if not value:
            return DecisionSnapshot(
                engine=self.name,
                model=None,
                input_text="",
            )

        started = time.perf_counter()
        result = self._get_router().predict(
            {"user_request": value},
            ASTA_DECISION_QUESTIONS,
            model=self.model,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        answers = result.get("answers") or {}
        routing = result.get("routing") or {}
        model = routing.get("model") or result.get("model")

        return DecisionSnapshot(
            engine=self.name,
            model=str(model) if model is not None else None,
            input_text=value,
            decisions={
                str(key): dict(answer)
                for key, answer in answers.items()
                if isinstance(answer, dict)
            },
            routing=dict(routing),
            latency_ms=elapsed_ms,
        )

    @classmethod
    def _normalize_model(cls, value: str | None) -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "multi": "multilingual",
            "ml": "multilingual",
            "laya-multilingual": "multilingual",
            "en": "english",
            "laya": "english",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls._SUPPORTED_MODELS:
            supported = ", ".join(sorted(cls._SUPPORTED_MODELS))
            raise ValueError(
                f"Unsupported Laya model '{value}'. Supported models: {supported}."
            )
        return normalized

    @staticmethod
    def _normalize_device(value: str | None) -> str | None:
        if not value:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"", "auto"}:
            return None
        return normalized

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}
