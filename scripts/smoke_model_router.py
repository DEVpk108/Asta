"""Real-model general -> coding -> general smoke. Does not run tools or load voice.

Close ASTA and any independently launched text-model servers first. This script
loads the ignored local .env and never kills an externally owned endpoint.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from core import Kernel
    from core.config import load_local_environment
    from ai.llm_provider import create_llm_provider
    from ai.model_router import RoutedLLMProvider

    load_local_environment()
    if os.getenv("ASTA_MODEL_ROUTER", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Enable ASTA_MODEL_ROUTER=1 and configure both GGUF paths first.")
    if os.getenv("ASTA_MODEL_MODE", "on_demand").strip().lower() != "on_demand":
        raise RuntimeError("Use scripts/smoke_concurrent_models.py for dual mode.")
    kernel = Kernel()
    try:
        provider = create_llm_provider(kernel=kernel)
        assert isinstance(provider, RoutedLLMProvider)
        if any(manager._server_is_ready() for manager in provider.runtime.managers.values()):
            raise RuntimeError("Close ASTA and independently launched text-model servers before this smoke test.")
        for expected, prompt in (
            ("general", "Reply with the word Ready."),
            ("coding", "Write a short Python function add(a, b) that returns their sum. Return code only."),
            ("general", "What is the capital of India? Answer briefly."),
        ):
            started = time.perf_counter()
            response = provider.generate_response(prompt)
            runtime = provider.runtime
            if not response or response.startswith("I couldn't use the requested"):
                raise RuntimeError("The model did not produce a usable response.")
            if runtime.active_profile != expected:
                raise RuntimeError(f"Expected {expected}, got {runtime.active_profile}.")
            if not runtime.managers[expected].owned:
                raise RuntimeError("Smoke inference must use an ASTA-owned server.")
            other = "coding" if expected == "general" else "general"
            if runtime.managers[other].running or runtime.managers[other]._server_is_ready():
                raise RuntimeError("The other text-model endpoint is still active.")
            print(json.dumps({
                "profile": expected,
                "model": str(provider.model).replace("\\", "/").rsplit("/", 1)[-1],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "reply": response,
            }, ensure_ascii=False), flush=True)
    finally:
        kernel.shutdown()
        runtime = getattr(kernel, "model_runtime", None)
        if runtime is not None and any(manager.owned for manager in runtime.managers.values()):
            raise RuntimeError("An owned model process could not be stopped.")
    print("ASTA_MODEL_ROUTER_SMOKE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model router smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
