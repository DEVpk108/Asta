from core.contracts import ToolRequest
from core.tools import (
    AuthorityPolicy,
    EchoTool,
    RiskLevel,
    ToolDispatcher,
    ToolRegistry,
)


def make_dispatcher(maximum_automatic_risk=RiskLevel.LOW):
    registry = ToolRegistry()
    registry.register(EchoTool())
    return ToolDispatcher(
        registry=registry,
        policy=AuthorityPolicy(
            maximum_automatic_risk=maximum_automatic_risk
        ),
    )


def test_echo_tool_dispatches_successfully():
    dispatcher = make_dispatcher()

    result = dispatcher.dispatch(
        ToolRequest(
            tool="echo",
            arguments={"text": "hello"},
            request_id="test-1",
        )
    )

    assert result.success is True
    assert result.tool == "echo"
    assert result.output == "hello"
    assert result.error is None
    assert result.metadata["request_id"] == "test-1"
    assert result.duration_seconds >= 0.0


def test_unknown_tool_returns_failure():
    dispatcher = make_dispatcher()

    result = dispatcher.dispatch(
        ToolRequest(
            tool="missing",
            arguments={},
            request_id="test-2",
        )
    )

    assert result.success is False
    assert result.tool == "missing"
    assert result.metadata["request_id"] == "test-2"
    assert "Unknown tool" in result.error


def test_tool_argument_validation_returns_failure():
    dispatcher = make_dispatcher()

    result = dispatcher.dispatch(
        ToolRequest(
            tool="echo",
            arguments={"text": 123},
            request_id="test-3",
        )
    )

    assert result.success is False
    assert "must be a string" in result.error


def test_policy_blocks_risk_above_automatic_limit():
    dispatcher = make_dispatcher(RiskLevel.MEDIUM)

    # Echo is LOW, so it remains automatically executable.
    result = dispatcher.dispatch(
        ToolRequest(
            tool="echo",
            arguments={"text": "safe"},
            request_id="test-4",
        )
    )

    assert result.success is True


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    registry.register(EchoTool())

    try:
        registry.register(EchoTool())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Duplicate tool registration should fail")
