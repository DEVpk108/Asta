import json
import threading
import time

from core.module import Module
from core.tools.approval import APPROVAL_EXPIRED_ERROR

from core.contracts import IntentResult, IntentType, ToolRequest, ToolResult
from core.tools.request_builder import ToolRequestBuilder
from core.tools.sandbox import is_sandbox_setup_error

APPROVED_RETRY_SECONDS = 900.0


class ToolRuntimeModule(Module):
    """EventBus adapter for tool dispatch and approval."""

    def __init__(self, kernel):
        super().__init__(
            name="ToolRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)
        # Approved commands that never ran because the sandbox wasn't ready:
        # after ASTA repairs it, exactly that command may run once more
        # without asking again.
        self._approved_retries: dict[tuple, float] = {}
        self._timer_lock = threading.RLock()
        self._timers = {}
        self._closing = False

    def initialize(self):
        self._closing = False
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        print("[Tools] Ready", flush=True)

    def shutdown(self):
        with self._timer_lock:
            self._closing = True
            timers = list(self._timers.values())
            self._timers.clear()
            self._approved_retries.clear()
        for timer in timers:
            timer.cancel()
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        print("[Tools] Stopped", flush=True)

    def on_tool_request(self, request):
        if self._closing:
            return
        if not isinstance(request, ToolRequest):
            print("[Tools] Ignoring invalid tool request.", flush=True)
            return

        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and not background.allow_request(request):
            self._emit_result(ToolResult(success=False, tool=request.tool,
                error="Background task cancelled or conflicting action blocked.",
                metadata={"background_conflict": True}), request=request)
            return
        overwrite = str((request.metadata or {}).get("confirm_overwrite") or "").strip()
        if overwrite:
            # Replacing an existing project file always asks first, whatever
            # the write tool's normal risk policy is.
            pending = self.kernel.approval_manager.request_approval(
                request=request,
                reason=(
                    f"Changing the test file {overwrite} was not requested; it needs approval."
                    if (request.metadata or {}).get("confirm_kind") == "test_edit"
                    else f"{overwrite} already exists with different content; replacing it needs approval."
                ),
            )
            self.event_bus.emit(
                "tool_confirmation_required",
                request=pending.request,
                reason=pending.reason,
            )
            self._schedule_expiry(pending.request.request_id)
            return

        result = self.kernel.tool_dispatcher.dispatch(
            request,
            confirmed=self._is_user_directed(request) or self._approved_retry(request),
        )

        if result.success or not result.metadata.get("requires_confirmation"):
            self._emit_result(result, request=request)
            return

        pending = self.kernel.approval_manager.request_approval(
            request=request,
            reason=result.error or "User confirmation is required.",
        )

        self.event_bus.emit(
            "tool_confirmation_required",
            request=pending.request,
            reason=pending.reason,
        )
        self._schedule_expiry(pending.request.request_id)

    def _schedule_expiry(self, request_id):
        ttl = float(getattr(self.kernel.approval_manager, "ttl_seconds", 0) or 0)
        if ttl <= 0:
            return
        with self._timer_lock:
            if self._closing or self.kernel.approval_manager.get(request_id) is None:
                return
            previous = self._timers.pop(request_id, None)
            if previous is not None:
                previous.cancel()
            timer = threading.Timer(ttl + 0.5, lambda: self._expire_approval(request_id, timer=timer))
            timer.daemon = True
            self._timers[request_id] = timer
            timer.start()

    def _cancel_expiry(self, request_id):
        with self._timer_lock:
            timer = self._timers.pop(request_id, None)
        if timer is not None:
            timer.cancel()

    def _expire_approval(self, request_id, *, timer=None):
        with self._timer_lock:
            if self._closing or (timer is not None and self._timers.get(request_id) is not timer):
                return
            self._timers.pop(request_id, None)
        manager = self.kernel.approval_manager
        pending = manager.get(request_id)
        if pending is None or manager.contains(request_id):
            return  # already answered, or still live
        print(f"[Tools] Approval expired (request_id={request_id}); step not run.", flush=True)
        background = getattr(self.kernel, "background_tasks", None)
        task_id = pending.request.metadata.get("task_id")
        if background is not None and background.owns(task_id) and not background.in_worker():
            background.schedule(lambda: self._confirmation_response(request_id, False, expired=True))
            return
        self._confirmation_response(request_id, False, expired=True)

    _USER_DIRECTED_TOOLS = frozenset({"computer.type_text", "computer.keypress"})

    @classmethod
    def _is_user_directed(cls, request) -> bool:
        """Typing the user's own spoken search query (and Enter) is consented.

        Only deterministic plans set this marker, only for keyboard steps
        whose content came verbatim from the utterance. Disable with
        ASTA_TRUST_USER_DIRECTED=0 to confirm every keyboard action.
        """
        import os

        if os.getenv("ASTA_TRUST_USER_DIRECTED", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        metadata = getattr(request, "metadata", None) or {}
        if metadata.get("user_directed") is not True:
            return False
        tool = getattr(request, "tool", "")
        if tool == "computer.hotkey":
            # Only the browser "focus address bar" shortcut is pre-approved.
            keys = (getattr(request, "arguments", None) or {}).get("keys") or []
            if isinstance(keys, str):
                keys = keys.replace("+", " ").split()
            return [str(k).strip().lower() for k in keys] == ["ctrl", "l"]
        return tool in cls._USER_DIRECTED_TOOLS

    def on_confirmation_response(self, request_id, approved):
        background = getattr(self.kernel, "background_tasks", None)
        pending = self.kernel.approval_manager.get(request_id) if isinstance(request_id, str) else None
        task_id = pending.request.metadata.get("task_id") if pending is not None else None
        if background is not None and background.owns(task_id) and not background.in_worker():
            background.schedule(lambda: self._confirmation_response(request_id, approved))
            return
        self._confirmation_response(request_id, approved)

    def _confirmation_response(self, request_id, approved, *, expired=False):
        if self._closing:
            return
        if not isinstance(request_id, str):
            self._emit_error(request_id, "Invalid request_id.")
            return

        if not isinstance(approved, bool):
            self._emit_error(request_id, "'approved' must be a boolean.")
            return

        self._cancel_expiry(request_id)
        manager = self.kernel.approval_manager
        pending = manager.take(request_id, expired_only=expired)
        if pending is None:
            return
        if manager._is_expired(pending):
            approved, expired = False, True
        if approved:
            request = pending.request

            background = getattr(self.kernel, "background_tasks", None)
            if background is not None and not background.allow_request(request):
                self._emit_result(ToolResult(success=False, tool=request.tool,
                    error="Background task cancelled or conflicting action blocked.",
                    metadata={"background_conflict": True}), request=request)
                return
            result = self.kernel.tool_dispatcher.dispatch(
                request,
                confirmed=True,
            )
            if not result.success and is_sandbox_setup_error(result.error):
                with self._timer_lock:
                    self._approved_retries[self._fingerprint(request)] = time.monotonic()
                    while len(self._approved_retries) > 64:
                        self._approved_retries.pop(next(iter(self._approved_retries)))
            self._emit_result(result, request=request)
            return

        rejection_metadata = {"request_id": request_id}
        if pending is not None:
            rejection_metadata.update(
                self._task_metadata(pending.request)
            )

        self._emit_result(
            ToolResult(
                success=False,
                tool="",
                error=(
                    APPROVAL_EXPIRED_ERROR
                    if expired
                    else "Tool execution rejected by user."
                ),
                metadata=rejection_metadata,
            )
        )

    @staticmethod
    def _fingerprint(request) -> tuple:
        metadata = getattr(request, "metadata", None) or {}
        try:
            arguments = json.dumps(request.arguments, sort_keys=True, default=str)
        except (TypeError, ValueError):
            arguments = repr(request.arguments)
        return (request.tool, arguments, metadata.get("task_id"), metadata.get("plan_step_id"))

    def _approved_retry(self, request) -> bool:
        """One rerun of an approved command that the sandbox couldn't start."""
        if not (getattr(request, "metadata", None) or {}).get("retry_after_setup"):
            return False
        with self._timer_lock:
            now = time.monotonic()
            for key, created in list(self._approved_retries.items()):
                if now - created > APPROVED_RETRY_SECONDS:
                    self._approved_retries.pop(key, None)
            created = self._approved_retries.pop(self._fingerprint(request), None)
        if created is None:
            return False
        print("[Tools] Rerunning the command you already approved, now that the sandbox is ready.", flush=True)
        return True

    def _emit_result(self, result, *, request=None):
        if request is not None:
            metadata = dict(result.metadata)
            metadata.update(self._task_metadata(request))
            result = ToolResult(
                success=result.success,
                tool=result.tool,
                output=result.output,
                error=result.error,
                duration_seconds=result.duration_seconds,
                metadata=metadata,
            )
        self.event_bus.emit("tool_result", result=result)

    @staticmethod
    def _task_metadata(request):
        metadata = {}
        for key in ("request_id", "task_id", "task_step", "plan_step_id", "planner", "completion_message"):
            if key == "request_id":
                metadata[key] = request.request_id
                continue
            if key in request.metadata:
                metadata[key] = request.metadata[key]
        return metadata

    def _emit_error(self, request_id, error, *, request=None):
        metadata = {"request_id": request_id}
        if request is not None:
            metadata.update(self._task_metadata(request))
        self._emit_result(
            ToolResult(
                success=False,
                tool="",
                error=error,
                metadata=metadata,
            )
        )
