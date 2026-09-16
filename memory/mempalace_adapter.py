"""MemPalace-backed long-term memory for A.S.T.A.

This adapter deliberately keeps MemPalace behind a small A.S.T.A.-owned
interface. A.S.T.A. should be able to replace the memory backend later without
changing the kernel or AI layer.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


class MemPalaceAdapter:
    """Store and retrieve A.S.T.A. memories through MemPalace."""

    def __init__(self, *, palace_path: str | None = None, wing: str = "asta"):
        self.palace_path = palace_path or os.getenv("ASTA_MEMPALACE_PATH")
        self.wing = wing
        self._collection = None
        self._stack = None
        self.available = False
        self.error: str | None = None

    def initialize(self) -> bool:
        """Load MemPalace lazily so A.S.T.A. can still run without it installed."""
        try:
            from mempalace.config import MempalaceConfig
            from mempalace.layers import MemoryStack
            from mempalace.palace import get_collection

            configured_path = self.palace_path or MempalaceConfig().palace_path
            self.palace_path = str(Path(configured_path).expanduser())

            self._collection = get_collection(self.palace_path, create=True)
            self._stack = MemoryStack(self.palace_path)
            self.available = True
            self.error = None
            return True
        except Exception as exc:  # MemPalace is an optional runtime dependency.
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def remember(self, *, role: str, text: str, session_id: str = "") -> bool:
        """Store one verbatim conversation message as a MemPalace drawer."""
        if not self.available or self._collection is None:
            return False

        value = str(text or "").strip()
        if not value:
            return False

        filed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source_file = f"asta_chat:{session_id or 'unknown'}"
        content = (
            f"A.S.T.A. conversation\n"
            f"session_id: {session_id or 'unknown'}\n"
            f"role: {role}\n"
            f"filed_at: {filed_at}\n\n"
            f"{value}"
        )

        try:
            from mempalace.ids import make_exchange_drawer_id

            drawer_id = make_exchange_drawer_id(
                self.wing,
                "conversation",
                source_file,
                filed_at,
                content,
            )
            self._collection.add(
                ids=[drawer_id],
                documents=[content],
                metadatas=[
                    {
                        "wing": self.wing,
                        "room": "conversation",
                        "source_file": source_file,
                        "chunk_index": 0,
                        "added_by": "asta",
                        "filed_at": filed_at,
                        "role": role,
                        "session_id": session_id or "unknown",
                    }
                ],
            )
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def delete_session(self, session_id: str) -> bool:
        """Remove all conversation memories belonging to one chat session."""
        if not self.available or self._collection is None:
            return False

        value = str(session_id or "").strip()
        if not value:
            return False

        try:
            self._collection.delete(where={"session_id": value})
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def search(self, query: str, *, n_results: int = 5) -> list[dict]:
        """Return raw semantic memory hits for the A.S.T.A. runtime."""
        if not self.available or self._stack is None:
            return []

        value = str(query or "").strip()
        if not value:
            return []

        try:
            return self._stack.l3.search_raw(
                value,
                wing=self.wing,
                n_results=max(1, int(n_results)),
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return []

    def context(self, query: str, *, n_results: int = 5, max_chars: int = 5000) -> str:
        """Format relevant memories into compact context for the AI model."""
        hits = self.search(query, n_results=n_results)
        if not hits:
            return ""

        lines = ["Relevant long-term memories from A.S.T.A.'s memory layer:"]
        total = len(lines[0])
        for index, hit in enumerate(hits, 1):
            text = " ".join(str(hit.get("text") or "").split())
            if not text:
                continue
            if len(text) > 900:
                text = text[:897].rstrip() + "..."
            line = f"[{index}] {text}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1

        return "\n".join(lines) if len(lines) > 1 else ""

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "palace_path": self.palace_path,
            "wing": self.wing,
            "error": self.error,
        }

    def close(self) -> None:
        self._collection = None
        self._stack = None
        self.available = False
