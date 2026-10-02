"""Local A.S.T.A. chat history."""

# ChatHistoryStore.delete_session lives in store.py. A duplicate override here
# referenced ``uuid`` without importing it, so deleting the active chat raised
# NameError after the DELETE statements had already run.
from .store import ChatHistoryStore


__all__ = ["ChatHistoryStore"]
