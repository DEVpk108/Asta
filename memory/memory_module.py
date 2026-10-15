"""A.S.T.A. long-term memory module.

MemPalace is the first backend. The module owns the integration boundary so the
rest of A.S.T.A. does not depend directly on MemPalace APIs.
"""

import logging

import queue
import threading

from core.module import Module

from .mempalace_adapter import MemPalaceAdapter


logger = logging.getLogger(__name__)

class MemoryModule(Module):
    """Persist conversation memory and prepare relevant recall context."""

    def __init__(self, kernel):
        super().__init__(
            name="MemoryModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.backend = MemPalaceAdapter()
        self._backend_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._closing = False
        self._deleted_sessions = set()
        # Embedding + persisting a memory takes tens of milliseconds. It used
        # to run inline in streaming callbacks and on the voice thread, delaying
        # speech. Full conversational replies now go through one writer.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None

    def _writer_loop(self):
        try:
            while True:
                item = self._write_queue.get()
                try:
                    if item is None:
                        return
                    self._remember(*item)
                except Exception as exc:
                    logger.warning(f"[Memory] Write failed: {type(exc).__name__}: {exc}")
                finally:
                    self._write_queue.task_done()
        finally:
            with self._backend_lock:
                self.backend.close()

    def _remember(self, role, text, session_id):
        with self._backend_lock:
            if session_id not in self._deleted_sessions:
                self.backend.remember(role=role, text=text, session_id=session_id)

    def _queue_write(self, role: str, text: str, session_id=None) -> None:
        if not text:
            return
        with self._state_lock:
            if self._closing:
                return
            item = (role, text, self._current_session_id() if session_id is None else session_id)
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._remember(*item)
            else:
                self._write_queue.put(item)

    def initialize(self):
        logger.info("[Memory] Initializing...")
        self._closing = False
        self._deleted_sessions.clear()
        self.backend.initialize()
        self.kernel.memory = self
        self.kernel.memory_context = ""

        if self.backend.available:
            logger.info(f"[Memory] MemPalace ready: {self.backend.palace_path}")
        else:
            logger.warning("[Memory] MemPalace unavailable; A.S.T.A. will run without long-term memory.")
            if self.backend.error:
                logger.info(f"[Memory] {self.backend.error}")

        if self.backend.available:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="MemoryWriter",
                daemon=True,
            )
            self._writer_thread.start()

        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("conversation_turn_completed", self.on_assistant_response)
        self.event_bus.subscribe("chat_session_changed", self.on_chat_session_changed)
        self.event_bus.subscribe("memory_request", self.on_memory_request)
        logger.info("[Memory] Ready")

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("conversation_turn_completed", self.on_assistant_response)
        self.event_bus.unsubscribe("chat_session_changed", self.on_chat_session_changed)
        self.event_bus.unsubscribe("memory_request", self.on_memory_request)
        with self._state_lock:
            if self._closing:
                return
            self._closing = True
            writer = self._writer_thread
            if writer is not None and writer.is_alive():
                self._write_queue.put(None)
        if writer is not None and writer.is_alive():
            if writer is not threading.current_thread():
                writer.join(timeout=5.0)
            # If still active, the writer's finally block owns backend close.
            if not writer.is_alive():
                self._writer_thread = None
        else:
            with self._backend_lock:
                self.backend.close()
        if getattr(self.kernel, "memory", None) is self:
            self.kernel.memory = None
        self.kernel.memory_context = ""
        logger.info("[Memory] Stopped accepting writes; queued writes drain before close.")

    def on_user_message(self, text):
        """Recall before storing the current message, so the query sees prior memory."""
        self.kernel.memory_context = self.recall(text, n_results=5)
        self._queue_write("user", text)

    def on_assistant_response(self, text, session_id=None):
        self._queue_write("assistant", text, session_id=session_id)

    def on_chat_session_changed(self, session_id):
        self.kernel.memory_context = ""

    def on_memory_request(self, intent=None, **_kwargs):
        """Handle the existing memory event without changing the intent router yet."""
        query = self._memory_query_from_intent(intent)
        if not query:
            return

        context = self.recall(query, n_results=8)
        self.kernel.memory_context = context
        self.event_bus.emit("memory_context_ready", query=query, context=context)

    def delete_session(self, session_id: str) -> bool:
        if self._closing:
            return False
        with self._backend_lock:
            if self._closing:
                return False
            self._deleted_sessions.add(session_id)
            deleted = self.backend.delete_session(session_id)
        if deleted:
            self.kernel.memory_context = ""
        return deleted

    def recall(self, query: str, *, n_results: int = 5) -> str:
        with self._backend_lock:
            if self._closing:
                return ""
            return self.backend.context(query, n_results=n_results)

    def status(self):
        with self._backend_lock:
            return self.backend.status()

    @property
    def available(self) -> bool:
        return bool(getattr(self.backend, "available", False))

    def _current_session_id(self) -> str:
        return str(getattr(self.kernel, "session_id", None) or
                   getattr(getattr(self.kernel, "chat_history", None), "session_id", ""))

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
