"""A.S.T.A. long-term memory module.

MemPalace is the first backend. The module owns the integration boundary so the
rest of A.S.T.A. does not depend directly on MemPalace APIs.
"""

from core.module import Module

from .mempalace_adapter import MemPalaceAdapter


class MemoryModule(Module):
    """Persist conversation memory and prepare relevant recall context."""

    def __init__(self, kernel):
        super().__init__(
            name="MemoryModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.backend = MemPalaceAdapter()

    def initialize(self):
        print("[Memory] Initializing...", flush=True)
        self.backend.initialize()
        self.kernel.memory = self
        self.kernel.memory_context = ""

        if self.backend.available:
            print(
                f"[Memory] MemPalace ready: {self.backend.palace_path}",
                flush=True,
            )
        else:
            print(
                "[Memory] MemPalace unavailable; A.S.T.A. will run without long-term memory.",
                flush=True,
            )
            if self.backend.error:
                print(f"[Memory] {self.backend.error}", flush=True)

        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.subscribe("memory_request", self.on_memory_request)
        print("[Memory] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.unsubscribe("memory_request", self.on_memory_request)
        self.backend.close()
        if getattr(self.kernel, "memory", None) is self:
            self.kernel.memory = None
        self.kernel.memory_context = ""
        print("[Memory] Stopped", flush=True)

    def on_user_message(self, text):
        """Recall before storing the current message, so the query sees prior memory."""
        self.kernel.memory_context = self.backend.context(text, n_results=5)
        self.backend.remember(
            role="user",
            text=text,
            session_id=self._current_session_id(),
        )

    def on_assistant_sentence(self, text):
        self.backend.remember(
            role="assistant",
            text=text,
            session_id=self._current_session_id(),
        )

    def on_memory_request(self, intent=None, **_kwargs):
        """Handle the existing memory event without changing the intent router yet."""
        query = self._memory_query_from_intent(intent)
        if not query:
            return

        context = self.backend.context(query, n_results=8)
        self.kernel.memory_context = context
        self.event_bus.emit("memory_context_ready", query=query, context=context)

    def delete_session(self, session_id: str) -> bool:
        """Forget all long-term conversation memories for one deleted chat."""
        deleted = self.backend.delete_session(session_id)
        if deleted:
            self.kernel.memory_context = ""
        return deleted

    def recall(self, query: str, *, n_results: int = 5) -> str:
        return self.backend.context(query, n_results=n_results)

    def status(self):
        return self.backend.status()

    def _current_session_id(self) -> str:
        """Use HUD's active chat session when available."""
        for module in getattr(self.kernel, "modules", []):
            store = getattr(module, "chat_history", None)
            session_id = getattr(store, "session_id", None)
            if session_id:
                return str(session_id)
        return ""

    @staticmethod
    def _memory_query_from_intent(intent) -> str:
        if intent is None:
            return ""

        entities = getattr(intent, "entities", {}) or {}
        for key in ("query", "text", "memory_query"):
            value = entities.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""
