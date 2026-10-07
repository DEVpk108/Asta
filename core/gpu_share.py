"""Share one small GPU between the chat LLM, Kokoro and the vision model.

On an 8 GB card the vision server (LFM2.5-VL) plus the chat LLM plus Kokoro
do not fit together; the driver then spills into system RAM and everything
crawls (LLM at a few tokens/s, Kokoro TTFA of seconds). Optional GPU users
register a release callback; the chat LLM asks them to step aside before it
generates a reply.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

_lock = threading.Lock()
# name -> (release callback, idle-since callback)
_releasers: dict[str, tuple[Callable[[], None], Callable[[], float | None]]] = {}


def register(name: str, release: Callable[[], None], idle_since: Callable[[], float | None]) -> None:
    with _lock:
        _releasers[name] = (release, idle_since)


def unregister(name: str) -> None:
    with _lock:
        _releasers.pop(name, None)


def release_idle(reason: str = "chat") -> list[str]:
    """Release optional GPU users idle for ASTA_GPU_RELEASE_IDLE_SECONDS."""
    if str(os.getenv("ASTA_GPU_RELEASE_FOR_CHAT", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return []
    try:
        min_idle = float(os.getenv("ASTA_GPU_RELEASE_IDLE_SECONDS", "5"))
    except ValueError:
        min_idle = 5.0
    now = time.monotonic()
    with _lock:
        items = list(_releasers.items())
    released = []
    for name, (release, idle_since) in items:
        try:
            since = idle_since()
            if since is None or now - since < min_idle:
                continue
            print(f"[GPU] Releasing {name} before {reason} to free VRAM.", flush=True)
            release()
            released.append(name)
        except Exception as exc:
            print(f"[GPU] Could not release {name}: {type(exc).__name__}: {exc}", flush=True)
    return released
