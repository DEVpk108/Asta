from core import Kernel
from core.contracts import ToolRequest
from core.tools import (
    OpenApplicationTool,
    RiskLevel,
    ToolRuntimeModule,
)


def make_kernel():
    kernel = Kernel(maximum_automatic_risk=RiskLevel.LOW)
    kernel.register_tool(OpenApplicationTool())

    runtime = ToolRuntimeModule(kernel)
    kernel.register_module(runtime)
    kernel.start()

    return kernel


def teardown(kernel):
    kernel.shutdown()


def test_medium_risk_request_creates_pending_approval():
    kernel = make_kernel()
    events = []
    kernel.event_bus.subscribe(
        "tool_confirmation_required",
        lambda request, reason: events.append((request, reason)),
    )

    request = ToolRequest(
        tool="system.open_application",
        arguments={"target": "https://example.com"},
        request_id="approval-1",
    )

    kernel.event_bus.emit("tool_request", request=request)

    assert len(events) == 1
    assert events[0][0] == request
    assert kernel.approval_manager.contains("approval-1")

    teardown(kernel)


def test_approved_request_executes_once():
    kernel = make_kernel()
    results = []
    kernel.event_bus.subscribe(
        "tool_result",
        lambda result: results.append(result),
    )

    request = ToolRequest(
        tool="system.open_application",
        arguments={"target": "https://example.com"},
        request_id="approval-2",
    )

    kernel.event_bus.emit("tool_request", request=request)
    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id="approval-2",
        approved=True,
    )

    assert len(results) == 1
    assert results[0].metadata["request_id"] == "approval-2"
    assert not kernel.approval_manager.contains("approval-2")

    # Approval is one-time; a second approval cannot execute the request.
    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id="approval-2",
        approved=True,
    )

    assert len(results) == 2
    assert results[1].success is False

    teardown(kernel)


def test_rejected_request_is_removed_without_execution():
    kernel = make_kernel()
    results = []
    kernel.event_bus.subscribe(
        "tool_result",
        lambda result: results.append(result),
    )

    request = ToolRequest(
        tool="system.open_application",
        arguments={"target": "https://example.com"},
        request_id="approval-3",
    )

    kernel.event_bus.emit("tool_request", request=request)
    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id="approval-3",
        approved=False,
    )

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].error == "Tool execution rejected by user."
    assert not kernel.approval_manager.contains("approval-3")

    teardown(kernel)
