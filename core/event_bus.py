"""
=========================================================
A.S.T.A. Cognitive OS
Event Bus
---------------------------------------------------------
A lightweight synchronous publish/subscribe bus. Modules
never import each other; they publish and subscribe to
named events on the kernel's bus instead.
=========================================================
"""

import logging
import threading


logger = logging.getLogger(__name__)


def _describe(callback):
    """Return a readable name without ever raising."""
    try:
        name = getattr(callback, "__qualname__", None)
    except Exception:
        name = None

    if not name:
        try:
            name = getattr(callback, "__name__", None)
        except Exception:
            name = None

    try:
        owner = getattr(callback, "__self__", None)
    except Exception:
        owner = None

    if name and owner is not None:
        try:
            owner_name = type(owner).__name__
        except Exception:
            owner_name = "object"
        # __qualname__ already carries the defining class, so keep only its
        # final component; otherwise the class name is printed twice.
        short = str(name).rsplit(".", 1)[-1]
        return f"{owner_name}.{short}"

    if name:
        return str(name)

    try:
        callback_type = type(callback).__name__
    except Exception:
        callback_type = "callable"

    try:
        representation = repr(callback)
    except Exception:
        representation = f"<{callback_type}>"

    return representation


class EventBus:
    """Simple synchronous publish/subscribe event bus.

    A.S.T.A. subscribes, unsubscribes and emits from several threads: the voice
    listen loop, the barge-in worker, the speech worker, the HUD transport
    accept/client threads and the main thread. The subscriber table is
    therefore guarded by a re-entrant lock.

    Callbacks run outside the lock, against a snapshot of the subscriber list,
    so a subscriber may safely emit, subscribe or unsubscribe while it is
    handling an event.
    """

    def __init__(self):
        # {event_type: [callback, ...]}
        self._subscribers = {}
        self._lock = threading.RLock()

    # -----------------------------------------------------
    # Subscription management
    # -----------------------------------------------------

    def subscribe(self, event_type, callback):
        """Register a callback for an event type. Duplicates are ignored."""
        with self._lock:
            callbacks = self._subscribers.setdefault(event_type, [])
            if callback not in callbacks:
                callbacks.append(callback)

    def unsubscribe(self, event_type, callback):
        """Remove a callback and drop the event key once it is empty."""
        with self._lock:
            callbacks = self._subscribers.get(event_type)
            if not callbacks:
                return

            if callback in callbacks:
                callbacks.remove(callback)

            if not callbacks:
                del self._subscribers[event_type]

    # -----------------------------------------------------
    # Event dispatch
    # -----------------------------------------------------

    def emit(self, event_type, *args, **kwargs):
        """Deliver an event to every current subscriber.

        One failing subscriber must not stop the others, so the error is logged
        with its traceback and dispatch continues.
        """
        with self._lock:
            callbacks = list(self._subscribers.get(event_type, ()))

        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception(
                    "[EventBus] Error in '%s' subscriber '%s'",
                    event_type,
                    _describe(callback),
                )

    # -----------------------------------------------------
    # Utilities
    # -----------------------------------------------------

    def clear(self):
        """Remove every subscription."""
        with self._lock:
            self._subscribers.clear()

    def has_subscribers(self, event_type):
        with self._lock:
            return bool(self._subscribers.get(event_type))

    def subscriber_count(self, event_type):
        with self._lock:
            return len(self._subscribers.get(event_type, ()))

    def registered_events(self):
        with self._lock:
            return list(self._subscribers.keys())
