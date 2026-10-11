from __future__ import annotations

import os
import threading
from typing import Callable, Protocol

_RUNTIME_LOCK = threading.Lock()

class LLMProvider(Protocol):
    system_prompt: str
    model: str | None

    def set_system_prompt(self, prompt: str) -> None:
        ...

    def warmup(self) -> bool:
        ...

    def reset_conversation(self) -> None:
        ...

    def shutdown(self) -> None:
        ...

    def generate_response(
        self,
        text: str,
        on_sentence: Callable[[str], None] | None = None,
        context: str | None = None,
    ) -> str:
        ...


def create_llm_provider(*, kernel=None, purpose="chat") -> LLMProvider:
    """Create the configured local LLM provider."""
    provider = os.getenv("ASTA_LLM_PROVIDER", "llama_cpp").strip().lower()

    if provider in {"llama_cpp", "llamacpp", "llama.cpp"}:
        if os.getenv("ASTA_MODEL_ROUTER", "0").strip().lower() in {"1", "true", "yes", "on"}:
            if kernel is None:
                raise ValueError("Model routing requires a kernel-scoped runtime.")
            from .model_router import ModelRuntime, RoutedLLMProvider

            with _RUNTIME_LOCK:
                runtime = getattr(kernel, "model_runtime", None)
                if runtime is None:
                    runtime = ModelRuntime(kernel)
                    kernel.model_runtime = runtime
            if os.getenv("ASTA_BACKGROUND_TASKS", "0").strip().lower() in {"1", "true", "yes", "on"}:
                from core.background_tasks import BackgroundTaskCoordinator
                with _RUNTIME_LOCK:
                    if getattr(kernel, "background_tasks", None) is None:
                        kernel.background_tasks = BackgroundTaskCoordinator(kernel)
            return RoutedLLMProvider(runtime, purpose=purpose)

        from .llama_cpp_engine import LlamaCppEngine

        return LlamaCppEngine()

    raise ValueError(
        f"Unsupported ASTA_LLM_PROVIDER '{provider}'. "
        "Currently supported provider: llama_cpp."
    )
