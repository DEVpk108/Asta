"""Small dependency-free SQLite store for A.S.T.A. conversation history."""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ChatHistoryStore:
    """Persist conversation history locally without coupling to the memory layer."""

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
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id);
            """
        )
        connection.commit()
        self._connection = connection

    def _ensure_current_session(self) -> None:
        if self._connection is None:
            return

        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                (self.session_id, self._now()),
            )

    def new_session(self) -> str:
        if self._connection is None:
            self.session_id = uuid.uuid4().hex
            return self.session_id

        new_id = uuid.uuid4().hex
        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                (new_id, self._now()),
            )
            self._connection.commit()
        self.session_id = new_id
        return new_id

    def switch_session(self, session_id: str) -> bool:
        value = str(session_id or "").strip()
        if not value or self._connection is None:
            return False

        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (value,),
            ).fetchone()
        if row is None:
            return False

        self.session_id = value
        return True

    def append(self, role: str, text: str) -> None:
        value = str(text or "").strip()
        if not value or self._connection is None:
            return

        normalized_role = "user" if role == "user" else "assistant"
        with self._lock:
            self._ensure_current_session()
            self._connection.execute(
                """
                INSERT INTO messages (session_id, role, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.session_id, normalized_role, value, self._now()),
            )
            self._connection.commit()

    def recent(self) -> list[dict[str, str]]:
        return self.messages_for_session(self.session_id)

    def messages_for_session(self, session_id: str, *, limit: int | None = None) -> list[dict[str, str]]:
        value = str(session_id or "").strip()
        if not value or self._connection is None:
            return []

        max_items = self.history_limit if limit is None else max(1, int(limit))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, text, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (value, max_items),
            ).fetchall()

        rows.reverse()
        return [
            {"role": role, "text": text, "created_at": created_at}
            for role, text, created_at in rows
        ]

    def sessions(self, *, limit: int = 50) -> list[dict[str, object]]:
        if self._connection is None:
            return []

        max_items = max(1, int(limit))
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT
                    s.id,
                    s.started_at,
                    (
                        SELECT m.text
                        FROM messages m
                        WHERE m.session_id = s.id AND m.role = 'user'
                        ORDER BY m.id ASC
                        LIMIT 1
                    ) AS title,
                    (
                        SELECT m.text
                        FROM messages m
                        WHERE m.session_id = s.id
                        ORDER BY m.id DESC
                        LIMIT 1
                    ) AS preview,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.session_id = s.id
                    ) AS message_count,
                    (
                        SELECT MAX(m.created_at)
                        FROM messages m
                        WHERE m.session_id = s.id
                    ) AS updated_at
                FROM sessions s
                WHERE EXISTS (
                    SELECT 1
                    FROM messages m
                    WHERE m.session_id = s.id
                )
                ORDER BY updated_at DESC, s.started_at DESC
                LIMIT ?
                """,
                (max_items,),
            ).fetchall()

        result = []
        for session_id, started_at, title, preview, message_count, updated_at in rows:
            clean_title = " ".join(str(title or "New conversation").split())
            if len(clean_title) > 48:
                clean_title = clean_title[:45].rstrip() + "..."
            clean_preview = " ".join(str(preview or "").split())
            if len(clean_preview) > 90:
                clean_preview = clean_preview[:87].rstrip() + "..."
            result.append(
                {
                    "id": session_id,
                    "title": clean_title,
                    "preview": clean_preview,
                    "message_count": int(message_count or 0),
                    "started_at": started_at,
                    "updated_at": updated_at or started_at,
                    "active": session_id == self.session_id,
                }
            )
        return result

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            with self._lock:
                connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
