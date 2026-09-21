from __future__ import annotations

import os
import time
from typing import Any

from .base import DecisionEngine
from .contracts import DecisionSnapshot
from .laya_schemas import ASTA_DECISION_QUESTIONS


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
        self.preload = self._env_bool(
            "ASTA_LAYA_PRELOAD",
            False if preload is None else preload,
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

        router = Router(**kwargs)

        # Router(preload=True) loads every checkpoint. A.S.T.A. deliberately
        # preloads only the selected model so multilingual remains the fast,
        # low-memory default while still avoiding first-request model loading.
        if self.preload:
            router.preload([self.model])

        self._router = router
        return self._router

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
