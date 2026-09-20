from __future__ import annotations

import os
from typing import Callable, Protocol


class LLMProvider(Protocol):
    system_prompt: str
    model: str | None

    def set_system_prompt(self, prompt: str) -> None:
        ...

    def warmup(self) -> bool:
        ...

    def reset_conversation(self) -> None:
        ...

    def generate_response(
        self,
        text: str,
        on_sentence: Callable[[str], None] | None = None,
        context: str | None = None,
    ) -> str:
        ...


def create_llm_provider() -> LLMProvider:
    """Create the configured local LLM provider."""
    provider = os.getenv("ASTA_LLM_PROVIDER", "llama_cpp").strip().lower()

    if provider in {"llama_cpp", "llamacpp", "llama.cpp"}:
        from .llama_cpp_engine import LlamaCppEngine

        return LlamaCppEngine()

    raise ValueError(
        f"Unsupported ASTA_LLM_PROVIDER '{provider}'. "
        "Currently supported provider: llama_cpp."
    )
