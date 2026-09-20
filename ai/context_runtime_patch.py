from __future__ import annotations

from core.context_builder import ContextBuilder


def apply_context_runtime_patch():
    """Feed the active runtime context into every normal model turn."""
    from ai.ai_module import AIModule

    if getattr(AIModule, "_asta_context_runtime_patch_applied", False):
        return

    original_generate_response = AIModule._generate_response

    def patched_generate_response(self, text, runtime_context=None):
        builder = getattr(self.kernel, "context_builder", None)
        if builder is None:
            builder = ContextBuilder(self.kernel)
            self.kernel.context_builder = builder

        context = runtime_context
        try:
            if not context:
                intent = self.kernel.intent_router.analyze(text)
                context = builder.build_prompt(text, intent)
        except Exception as exc:  # context enrichment must not break chat
            print(
                f"[Context] Failed to build runtime context: {type(exc).__name__}: {exc}",
                flush=True,
            )
            context = None

        return original_generate_response(self, text, runtime_context=context)

    AIModule._generate_response = patched_generate_response
    AIModule._asta_context_runtime_patch_applied = True
