import time
from typing import Any

from core.contracts import (
    ToolRequest,
    ToolResult,
)

from core.tools.policy import (
    AuthorityPolicy,
)

from core.tools.registry import (
    ToolRegistry,
)


class ToolDispatcher:
    """Execute ToolRequests through registered tools and policy."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: AuthorityPolicy | None = None,
    ):
        self.registry = registry
        self.policy = policy if policy is not None else AuthorityPolicy()

    def dispatch(
        self,
        request: ToolRequest,
        *,
        confirmed: bool = False,
    ) -> ToolResult:
        """Dispatch one request.

        ``confirmed=True`` is only intended for requests that have already
        passed through A.S.T.A.'s ApprovalManager.
        """
        start = time.perf_counter()

        try:
            tool = self.registry.get(request.tool)
        except KeyError:
            return self._failure(
                request=request,
                start=start,
                error=f"Unknown tool: {request.tool}",
            )

        try:
            authorization = self.policy.authorize(
                tool.definition,
                confirmed=confirmed,
            )
        except Exception as exc:
            return self._failure(
                request=request,
                start=start,
                error=(
                    f"Authorization error: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not authorization.allowed:
            return self._failure(
                request=request,
                start=start,
                error=authorization.reason,
                metadata={
                    "requires_confirmation": authorization.requires_confirmation,
                },
            )

        try:
            result = tool.execute(request)
        except Exception as exc:
            return self._failure(
                request=request,
                start=start,
                error=f"{type(exc).__name__}: {exc}",
            )

        elapsed = time.perf_counter() - start

        return ToolResult(
            success=result.success,
            tool=result.tool,
            output=result.output,
            error=result.error,
            duration_seconds=elapsed,
            metadata={
                **result.metadata,
                "request_id": request.request_id,
            },
        )

    @staticmethod
    def _failure(
        *,
        request: ToolRequest,
        start: float,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        elapsed = time.perf_counter() - start

        return ToolResult(
            success=False,
            tool=request.tool,
            error=error,
            duration_seconds=elapsed,
            metadata={
                "request_id": request.request_id,
                **(metadata or {}),
            },
        )
