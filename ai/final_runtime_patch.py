"""Compatibility entry point; native AI/HUD code owns post-response capture."""


def apply_final_runtime_patch():
    """Kept for older callers; intentionally does not replace any methods."""
    return None
