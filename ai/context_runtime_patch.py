"""Retired context-wrapper compatibility entry point.

AIModule's native generation now owns context preparation. Keeping this hook
does not require callers to know which other runtime patches have been applied.
"""


def apply_context_runtime_patch():
    """Compatibility no-op; native generation prepares context exactly once."""
    return None
