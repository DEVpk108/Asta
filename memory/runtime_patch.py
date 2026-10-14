"""Retired memory-wrapper compatibility entry point.

MemoryModule recalls into kernel.memory_context; ContextBuilder decides whether
to include it in a prompt. The former wrapper only forwarded arguments to the
existing generation method and did not inject or persist memory.
"""


def apply_memory_runtime_patch():
    """Compatibility no-op for older startup code; never mutate AIModule.

    Main no longer imports or calls this hook. Keep the callable so an older
    launcher can invoke it without replacing the current generation pipeline.
    """
    return None
