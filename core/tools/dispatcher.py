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
    """
    Executes authorized ToolRequests through registered tools.

    Responsibilities:
        - validate tool existence
        - authorize execution
        - invoke the tool
        - measure execution time
        - normalize failures
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: AuthorityPolicy | None = None,
    ):
        self.registry = registry

        self.policy = (
            policy
            if policy is not None
            else AuthorityPolicy()
        )

    def dispatch(
        self,
        request: ToolRequest,
    ) -> ToolResult:

        start = time.perf_counter()

        try:
            tool = self.registry.get(
                request.tool
            )

        except KeyError as exc:

            return self._failure(
                request=request,
                start=start,
                error=(
            f"Unknown tool: {request.tool}"
                ),
            )

        # -----------------------------------------------------
        # Authorization
        # -----------------------------------------------------

        try:
            authorization = (
                self.policy.authorize(
                    tool.definition
                )
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
                    "requires_confirmation": (
                        authorization
                        .requires_confirmation
                    )
                },
            )

        # -----------------------------------------------------
        # Execute
        # -----------------------------------------------------

        try:
            result = tool.execute(
                request
            )

        except Exception as exc:

            return self._failure(
                request=request,
                start=start,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ToolResult is supposed to already contain the
        # semantic result. The dispatcher owns final timing.
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

    # =========================================================
    # Internal helpers
    # =========================================================

    @staticmethod
    def _failure(
        *,
        request: ToolRequest,
        start: float,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:

        elapsed = (
            time.perf_counter()
            - start
        )

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