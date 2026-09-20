"""Backward-compatible import shim.

The active A.S.T.A. runtime uses llama.cpp through ``ai.llama_cpp_engine``.
"""

from .llama_cpp_engine import LlamaCppEngine

AIEngine = LlamaCppEngine

__all__ = ["AIEngine", "LlamaCppEngine"]
