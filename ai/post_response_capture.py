"""Owned response-to-screenshot sequencing; no import-time class mutation.

The HUD acknowledgement is a paint barrier, not proof the HUD is visible on
an unoccluded desktop. A timeout preserves headless operation and is labelled
as a fallback in request metadata. All tool dispatch still uses ToolRuntime.
"""
import re
import threading
import uuid

_POST_RESPONSE_SCREENSHOT_PATTERN = re.compile(
    r"^(?P<prompt>.+?)\s*(?:,?\s+)(?:and\s+then|then|after that|followed by|and)\s+"
    r"(?:please\s+)?"
    r"(?P<action>"
    r"(?:take(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot))"
    r"|(?:capture(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot))"
    r"|(?:screenshot|screen\s*shot)"
    r")$",
    re.IGNORECASE,
)


def extract_prompt(text):
    if not isinstance(text, str):
        return None

    normalized = " ".join(text.strip().split())
    match = _POST_RESPONSE_SCREENSHOT_PATTERN.match(normalized.rstrip(" .!?;:"))
    if not match:
        return None

    prompt = match.group("prompt").strip(" ,.!?;:")
    if not prompt:
        return None
    return prompt


class PostResponseCapture:
    """One cancellable pending response; ack and timeout compete under one lock."""

    ACK_TIMEOUT = 1.5
    SETTLE_DELAY = 0.20

    def __init__(self, event_bus, timer_factory=threading.Timer):
        self.bus = event_bus
        self._timer_factory = timer_factory
        self._condition = threading.Condition(threading.RLock())
        self._pending = None
        self._closed = False
        self._dispatch_threads = {}
        self.bus.subscribe("hud_rendered", self.on_rendered)

    def begin(self, request):
        self.cancel()
        with self._condition:
            if self._closed:
                return None
            token = uuid.uuid4().hex
            self._pending = {"id": token, "request": request, "phase": "generating", "timer": None}
        self.bus.emit("capture_response_begin", response_id=token)
        return token

    def complete(self, token, response):
        if not response:
            self.cancel(token)
            return
        with self._condition:
            pending = self._pending
            if self._closed or not pending or pending["id"] != token or pending["phase"] != "generating":
                return
            pending["phase"] = "waiting"
            self._arm(pending, self.ACK_TIMEOUT, lambda: self._settle(token, "timeout_fallback"))
        self.bus.emit("capture_response_ready", response_id=token, text=response)

    def _arm(self, pending, delay, callback):
        # Called under the condition; Timer.start never invokes callback inline.
        timer = self._timer_factory(delay, callback)
        timer.daemon = True
        pending["timer"] = timer
        timer.start()

    def on_rendered(self, response_id):
        if isinstance(response_id, str):
            self._settle(response_id, "hud_acknowledged")

    def _settle(self, token, reason):
        with self._condition:
            pending = self._pending
            if self._closed or not pending or pending["id"] != token or pending["phase"] != "waiting":
                return
            pending["timer"].cancel()
            pending["phase"] = "settling"
            pending["request"].metadata["capture_render_status"] = reason
            pending["request"].metadata["capture_response_id"] = token
            self._arm(pending, self.SETTLE_DELAY, lambda: self._dispatch(token))

    def _dispatch(self, token):
        with self._condition:
            pending = self._pending
            if self._closed or not pending or pending["id"] != token or pending["phase"] != "settling":
                return
            self._pending = None
            thread_id = threading.get_ident()
            self._dispatch_threads[thread_id] = self._dispatch_threads.get(thread_id, 0) + 1
        try:
            # Never invoke the synchronous event bus while holding our lock.
            self.bus.emit("capture_response_cancel", response_id=token)
            with self._condition:
                if self._closed:
                    return
            self.bus.emit("tool_request", request=pending["request"])
        finally:
            with self._condition:
                self._dispatch_threads[thread_id] -= 1
                if not self._dispatch_threads[thread_id]:
                    del self._dispatch_threads[thread_id]
                self._condition.notify_all()

    def cancel(self, token=None):
        with self._condition:
            pending = self._pending
            if not pending or (token is not None and pending["id"] != token):
                return
            self._pending = None
            if pending["timer"] is not None:
                pending["timer"].cancel()
        self.bus.emit("capture_response_cancel", response_id=pending["id"])

    def close(self):
        with self._condition:
            self._closed = True
        self.cancel()
        self.bus.unsubscribe("hud_rendered", self.on_rendered)
        # A callback already claimed before shutdown may finish, but none may
        # dispatch after close returns. Reentrant shutdown must not self-wait.
        with self._condition:
            while any(thread_id != threading.get_ident() for thread_id in self._dispatch_threads):
                self._condition.wait()
