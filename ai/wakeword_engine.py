"""Deprecated import path for the canonical voice wakeword engine.

The runtime uses voice.wakeword_engine. Do not maintain a second detector here.
"""

def __getattr__(name):
    if name in {"WakeWordEngine", "DEFAULT_MODELS", "MODEL_DIR", "ROOT"}:
        from voice import wakeword_engine
        return getattr(wakeword_engine, name)
    raise AttributeError(name)
