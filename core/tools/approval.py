from dataclasses import dataclass

from core.contracts import ToolRequest


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """A tool request waiting for explicit user approval."""

    request: ToolRequest
    reason: str


class ApprovalManager:
    """Track tool requests that require explicit approval.

    Approval is intentionally tied to the exact request ID. A request that
    was never registered as pending cannot be approved through this manager.
    """

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}

    def request_approval(
        self,
        request: ToolRequest,
        reason: str,
    ) -> PendingApproval:
        pending = PendingApproval(
            request=request,
            reason=reason,
        )
        self._pending[request.request_id] = pending
        return pending

    def get(self, request_id: str) -> PendingApproval | None:
        return self._pending.get(request_id)

    def approve(self, request_id: str) -> ToolRequest | None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            return None
        return pending.request

    def reject(self, request_id: str) -> bool:
        return self._pending.pop(request_id, None) is not None

    def contains(self, request_id: str) -> bool:
        return request_id in self._pending

    def list_pending(self) -> tuple[PendingApproval, ...]:
        return tuple(self._pending.values())

    def clear(self) -> None:
        self._pending.clear()
