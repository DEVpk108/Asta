"""
=========================================================
A.S.T.A. Cognitive OS
Event Bus
---------------------------------------------------------
A lightweight publish/subscribe messaging system used by
the A.S.T.A. Kernel to allow modules to communicate
without direct dependencies.
=========================================================
"""


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self):
        # {event_name: [callback1, callback2, ...]}
        self._subscribers = {}

    # -----------------------------------------------------
    # Subscription Management
    # -----------------------------------------------------

    def subscribe(self, event_type: str, callback):
        """
        Register a callback for an event.

        Example:
            event_bus.subscribe("assistant_response", on_response)
        """
        callbacks = self._subscribers.setdefault(event_type, [])

        # Prevent duplicate subscriptions
        if callback not in callbacks:
            callbacks.append(callback)

    def unsubscribe(self, event_type: str, callback):
        """
        Remove a callback from an event.
        """
        callbacks = self._subscribers.get(event_type)

        if not callbacks:
            return

        if callback in callbacks:
            callbacks.remove(callback)

        # Remove empty event lists
        if not callbacks:
            del self._subscribers[event_type]

    # -----------------------------------------------------
    # Event Dispatch
    # -----------------------------------------------------

    def emit(self, event_type: str, *args, **kwargs):
        """
        Emit an event to all subscribers.

        Example:
            event_bus.emit(
                "assistant_response",
                text="Hello"
            )
        """

        callbacks = self._subscribers.get(event_type, [])

        # Iterate over a copy in case subscribers modify
        # the list while events are being processed.
        for callback in callbacks[:]:
            try:
                callback(*args, **kwargs)

            except Exception as e:
                # For now keep it simple.
                # Later this will be routed to ASTA Logger.
                print(
                    f"[EventBus] Error in '{event_type}' "
                    f"subscriber '{callback.__name__}': {e}"
                )

    # -----------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------

    def clear(self):
        """Remove all subscribers."""
        self._subscribers.clear()

    def has_subscribers(self, event_type: str) -> bool:
        """Return True if the event has subscribers."""
        return event_type in self._subscribers

    def subscriber_count(self, event_type: str) -> int:
        """Return the number of subscribers for an event."""
        return len(self._subscribers.get(event_type, []))

    def registered_events(self):
        """Return a list of all registered event names."""
        return list(self._subscribers.keys())