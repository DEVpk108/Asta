"""Local A.S.T.A. chat history."""

from .store import ChatHistoryStore as _ChatHistoryStore


class ChatHistoryStore(_ChatHistoryStore):
    """Chat history store with session deletion support."""

    def delete_session(self, session_id: str) -> bool:
        """Delete one chat session and its messages from local chat history."""
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

            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (value,))
            self._connection.execute("DELETE FROM sessions WHERE id = ?", (value,))

            if value == self.session_id:
                self.session_id = uuid.uuid4().hex
                self._connection.execute(
                    "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
                    (self.session_id, self._now()),
                )

            self._connection.commit()
        return True


__all__ = ["ChatHistoryStore"]
