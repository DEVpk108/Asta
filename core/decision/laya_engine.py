from __future__ import annotations

import os
import time
from typing import Any

from core.contracts.action import ActionDecision, ActionType

from .base import DecisionEngine
from .contracts import DecisionSnapshot
from .laya_schemas import ASTA_DECISION_QUESTIONS
from .laya_action_schemas import build_action_questions


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
        questions = build_action_questions(names)

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
