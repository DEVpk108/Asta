"""Runtime bridge that injects recalled MemPalace context into A.S.T.A. generation."""

from ai.ai_module import AIModule


_PATCHED = False
_ORIGINAL_GENERATE_RESPONSE = None


def apply_memory_runtime_patch():
    """Add memory context without coupling AIModule directly to MemPalace."""
    global _PATCHED, _ORIGINAL_GENERATE_RESPONSE

    if _PATCHED:
        return

    original = AIModule._generate_response
    _ORIGINAL_GENERATE_RESPONSE = original

    def _generate_response_with_memory(self, text, runtime_context=None):
        """Preserve the context pipeline without persisting memory into chat history."""
        return original(self, text, runtime_context=runtime_context)

    AIModule._generate_response = _generate_response_with_memory
    _PATCHED = True
