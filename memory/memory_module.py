"""A.S.T.A. long-term memory module.

MemPalace is the first backend. The module owns the integration boundary so the
rest of A.S.T.A. does not depend directly on MemPalace APIs.
"""

import queue
import threading

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
        # Embedding + persisting a memory takes tens of milliseconds. It used
        # to run inline in the LLM's per-sentence streaming callback and on the
        # voice thread, delaying speech. Writes now go through one worker.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None

    def _writer_loop(self):
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                role, text, session_id = item
                self.backend.remember(role=role, text=text, session_id=session_id)
            except Exception as exc:
                print(
                    f"[Memory] Write failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                self._write_queue.task_done()

    def _queue_write(self, role: str, text: str) -> None:
        if not text:
            return
        item = (role, text, self._current_session_id())
        if self._writer_thread is None or not self._writer_thread.is_alive():
            # Not initialized (e.g. direct use in tests): write synchronously.
            self.backend.remember(role=item[0], text=item[1], session_id=item[2])
            return
        self._write_queue.put(item)

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

        if self.backend.available:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="MemoryWriter",
                daemon=True,
            )
            self._writer_thread.start()

        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.subscribe("memory_request", self.on_memory_request)
        print("[Memory] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.unsubscribe("memory_request", self.on_memory_request)
        writer = self._writer_thread
        if writer is not None and writer.is_alive():
            # Flush pending writes before closing the backend.
            self._write_queue.put(None)
            writer.join(timeout=5.0)
        self._writer_thread = None
        self.backend.close()
        if getattr(self.kernel, "memory", None) is self:
            self.kernel.memory = None
        self.kernel.memory_context = ""
        print("[Memory] Stopped", flush=True)

    def on_user_message(self, text):
        """Recall before storing the current message, so the query sees prior memory."""
        self.kernel.memory_context = self.backend.context(text, n_results=5)
        self._queue_write("user", text)

    def on_assistant_sentence(self, text):
        self._queue_write("assistant", text)

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

    @property
    def available(self) -> bool:
        return bool(getattr(self.backend, "available", False))

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
