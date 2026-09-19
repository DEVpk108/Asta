from core import Kernel
from core.contracts import ToolDefinition, ToolRequest
from core.tools import (
    AuthorityManager,
    AuthorityMode,
    AuthorityPolicy,
    EchoTool,
    RiskLevel,
)


def definition(name="test_tool", risk="low", requires_confirmation=False):
    return ToolDefinition(
        name=name,
        description="Authority test capability.",
        input_schema={"type": "object"},
        risk_level=risk,
        requires_confirmation=requires_confirmation,
    )


def test_default_policy_remains_the_fallback():
    manager = AuthorityManager(
        policy=AuthorityPolicy(maximum_automatic_risk=RiskLevel.MEDIUM)
    )

    authorization = manager.authorize(definition(risk="medium"))

    assert authorization.allowed is True
    assert authorization.requires_confirmation is False


def test_grant_can_override_default_high_risk_confirmation():
    manager = AuthorityManager(
        policy=AuthorityPolicy(maximum_automatic_risk=RiskLevel.MEDIUM)
    )
    manager.grant("test_tool", reason="Trusted local capability.")

    authorization = manager.authorize(definition(risk="high"))

    assert authorization.allowed is True
    assert authorization.requires_confirmation is False
    assert manager.get_rule("test_tool").mode is AuthorityMode.AUTO


def test_confirm_rule_requires_confirmation_until_explicitly_confirmed():
    manager = AuthorityManager()
    manager.require_confirmation("test_tool")

    blocked = manager.authorize(definition())
    confirmed = manager.authorize(definition(), confirmed=True)

    assert blocked.allowed is False
    assert blocked.requires_confirmation is True
    assert confirmed.allowed is True


def test_deny_rule_blocks_even_low_risk_capability():
    manager = AuthorityManager()
    manager.deny("test_tool", reason="Blocked for this session.")

    authorization = manager.authorize(definition())

    assert authorization.allowed is False
    assert authorization.requires_confirmation is False
    assert "Blocked for this session." in authorization.reason


def test_rules_are_serializable_for_future_persistence():
    manager = AuthorityManager()
    manager.grant("one")
    manager.require_confirmation("two")
    manager.deny("three")

    snapshot = manager.snapshot()

    assert snapshot == (
        {"tool": "one", "mode": "auto", "reason": ""},
        {"tool": "two", "mode": "confirm", "reason": ""},
        {"tool": "three", "mode": "deny", "reason": ""},
    )


def test_kernel_owns_authority_manager_and_dispatch_uses_it():
    kernel = Kernel()
    kernel.register_tool(EchoTool())
    kernel.authority_manager.deny("echo", reason="No echo capability.")

    result = kernel.tool_dispatcher.dispatch(
        ToolRequest(
            tool="echo",
            arguments={"text": "hello"},
            request_id="authority-test",
        )
    )

    assert result.success is False
    assert "No echo capability." in result.error
    assert result.metadata["requires_confirmation"] is False
