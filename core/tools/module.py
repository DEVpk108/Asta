from core.module import Module

from core.contracts import IntentResult, IntentType, ToolRequest, ToolResult
from core.tools.request_builder import ToolRequestBuilder


class ToolRuntimeModule(Module):
    """EventBus adapter for tool dispatch and approval."""

    def __init__(self, kernel):
        super().__init__(
            name="ToolRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)

    def initialize(self):
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        print("[Tools] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        print("[Tools] Stopped", flush=True)

    def on_tool_request(self, request):
        if not isinstance(request, ToolRequest):
            print("[Tools] Ignoring invalid tool request.", flush=True)
            return

        result = self.kernel.tool_dispatcher.dispatch(
            request,
            confirmed=self._is_user_directed(request),
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
        if not isinstance(request_id, str):
            self._emit_error(request_id, "Invalid request_id.")
            return

        if not isinstance(approved, bool):
            self._emit_error(request_id, "'approved' must be a boolean.")
            return

        if approved:
            request = self.kernel.approval_manager.approve(request_id)

            if request is None:
                self._emit_error(
                    request_id,
                    "No pending tool request exists for this request_id.",
                )
                return

            result = self.kernel.tool_dispatcher.dispatch(
                request,
                confirmed=True,
            )
            self._emit_result(result, request=request)
            return

        pending = self.kernel.approval_manager.get(request_id)
        rejected = self.kernel.approval_manager.reject(request_id)

        if not rejected:
            self._emit_error(
                request_id,
                "No pending tool request exists for this request_id.",
            )
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
                error="Tool execution rejected by user.",
                metadata=rejection_metadata,
            )
        )

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
