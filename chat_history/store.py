"""Small dependency-free SQLite store for A.S.T.A. conversation history."""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ChatHistoryStore:
    """Persist conversation messages locally without coupling to the memory layer."""

    def __init__(self, db_path: str | Path | None = None, *, history_limit: int = 200):
        root = Path(__file__).resolve().parents[1]
        configured = db_path or os.getenv("ASTA_CHAT_HISTORY_DB")
        self.db_path = Path(configured) if configured else root / "data" / "chat_history.db"
        self.history_limit = max(1, int(history_limit))
        self._lock = threading.Lock()
        self._connection = None
        self.session_id = uuid.uuid4().hex

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_created_at
                ON messages(created_at);
            """
        )
        connection.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (self.session_id, self._now()),
        )
        connection.commit()
        self._connection = connection

    def append(self, role: str, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        if self._connection is None:
            return

        normalized_role = "user" if role == "user" else "assistant"
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO messages (session_id, role, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.session_id, normalized_role, value, self._now()),
            )
            self._connection.commit()

    def recent(self) -> list[dict[str, str]]:
        if self._connection is None:
            return []

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, text, created_at
                FROM messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.history_limit,),
            ).fetchall()

        rows.reverse()
        return [
            {"role": role, "text": text, "created_at": created_at}
            for role, text, created_at in rows
        ]

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            with self._lock:
                connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
