from core.module import Module

from core.contracts import IntentResult, IntentType, ToolRequest, ToolResult
from core.tools.request_builder import ToolRequestBuilder


class ToolRuntimeModule(Module):
    """EventBus adapter for tool dispatch, approval, and command sequences."""

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

        result = self.kernel.tool_dispatcher.dispatch(request)

        if result.success or not result.metadata.get("requires_confirmation"):
            self._emit_result(result)
            if result.success:
                self._continue_sequence(request)
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
            self._emit_result(result)
            if result.success:
                self._continue_sequence(request)
            return

        rejected = self.kernel.approval_manager.reject(request_id)

        if not rejected:
            self._emit_error(
                request_id,
                "No pending tool request exists for this request_id.",
            )
            return

        self._emit_result(
            ToolResult(
                success=False,
                tool="",
                error="Tool execution rejected by user.",
                metadata={"request_id": request_id},
            )
        )

    def _continue_sequence(self, request: ToolRequest):
        sequence = request.metadata.get("sequence")
        index = request.metadata.get("sequence_index")

        if not isinstance(sequence, list) or not isinstance(index, int):
            return
        if index < 0 or index + 1 >= len(sequence):
            return

        next_index = index + 1
        next_entities = sequence[next_index]
        if not isinstance(next_entities, dict):
            self._emit_error(
                request.request_id,
                "Invalid compound command step.",
            )
            return

        next_intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=float(request.metadata.get("intent_confidence", 0.98)),
            normalized_text=str(request.metadata.get("normalized_text", "")),
            entities=dict(next_entities),
            requires_tools=True,
            classifier=str(request.metadata.get("classifier", "rules")),
        )

        try:
            next_request = self.tool_request_builder.build(next_intent)
        except ValueError as exc:
            self._emit_error(request.request_id, str(exc))
            return

        next_request.metadata["sequence"] = [dict(command) for command in sequence]
        next_request.metadata["sequence_index"] = next_index

        print(
            f"[Tools] Continuing compound command: "
            f"step={next_index + 1}/{len(sequence)} tool={next_request.tool}",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=next_request)

    def _emit_result(self, result):
        self.event_bus.emit("tool_result", result=result)

    def _emit_error(self, request_id, error):
        self._emit_result(
            ToolResult(
                success=False,
                tool="",
                error=error,
                metadata={"request_id": request_id},
            )
        )
