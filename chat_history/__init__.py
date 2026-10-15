"""Local A.S.T.A. chat history."""

# ChatHistoryStore.delete_session lives in store.py. A duplicate override here
# referenced ``uuid`` without importing it, so deleting the active chat raised
# NameError after the DELETE statements had already run.
from .store import ChatHistoryStore


__all__ = ["ChatHistoryStore", "shared_history"]

# Construction is also safe for small standalone module/test kernels.
import threading
_creation_lock = threading.RLock()


def shared_history(kernel):
    with _creation_lock:
        store = getattr(kernel, "chat_history", None)
        if store is None:
            store = kernel.chat_history = ChatHistoryStore()
        return store
