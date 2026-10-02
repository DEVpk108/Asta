import os
import threading
import time
from dataclasses import dataclass, field

from core.contracts import ToolRequest


DEFAULT_APPROVAL_TTL_SECONDS = 30.0


def _default_ttl_seconds() -> float:
    try:
        value = float(
            os.getenv(
                "ASTA_APPROVAL_TTL_SECONDS",
                str(DEFAULT_APPROVAL_TTL_SECONDS),
            )
        )
    except (TypeError, ValueError):
        value = DEFAULT_APPROVAL_TTL_SECONDS
    return max(1.0, value)


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """A tool request waiting for explicit user approval."""

    request: ToolRequest
    reason: str
    created_at: float = field(default_factory=time.monotonic)


class ApprovalManager:
    """Track tool requests that require explicit approval.

    Approval is intentionally tied to the exact request ID. A request that
    was never registered as pending cannot be approved through this manager.

    Pending approvals expire after ``ttl_seconds``. A stale request must never
    be executable by an unrelated "yes" spoken minutes later, so expired
    requests are hidden from ``list_pending`` and cannot be approved. They stay
    retrievable through ``list_expired``/``get`` so the runtime can reject
    them cleanly and fail the owning task.
    """

    def __init__(self, ttl_seconds: float | None = None):
        self.ttl_seconds = (
            _default_ttl_seconds()
            if ttl_seconds is None
            else max(0.0, float(ttl_seconds))
        )
        self._pending: dict[str, PendingApproval] = {}
        self._lock = threading.RLock()

    def _is_expired(self, pending: PendingApproval, now: float | None = None) -> bool:
        if self.ttl_seconds <= 0:
            return False
        current = time.monotonic() if now is None else now
        return current - pending.created_at > self.ttl_seconds

    def request_approval(
        self,
        request: ToolRequest,
        reason: str,
    ) -> PendingApproval:
        pending = PendingApproval(
            request=request,
            reason=reason,
        )
        with self._lock:
            self._pending[request.request_id] = pending
        return pending

    def get(self, request_id: str) -> PendingApproval | None:
        with self._lock:
            return self._pending.get(request_id)

    def approve(self, request_id: str) -> ToolRequest | None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None or self._is_expired(pending):
            return None
        return pending.request

    def reject(self, request_id: str) -> bool:
        with self._lock:
            return self._pending.pop(request_id, None) is not None

    def contains(self, request_id: str) -> bool:
        with self._lock:
            pending = self._pending.get(request_id)
        return pending is not None and not self._is_expired(pending)

    def list_pending(self) -> tuple[PendingApproval, ...]:
        now = time.monotonic()
        with self._lock:
            return tuple(
                pending
                for pending in self._pending.values()
                if not self._is_expired(pending, now)
            )

    def list_expired(self) -> tuple[PendingApproval, ...]:
        now = time.monotonic()
        with self._lock:
            return tuple(
                pending
                for pending in self._pending.values()
                if self._is_expired(pending, now)
            )

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
