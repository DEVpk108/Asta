from core import Kernel
from core.contracts import ToolDefinition, ToolRequest
from core.tools import (
    AuthorityManager,
    AuthorityMode,
    AuthorityPolicy,
    AuthorityStore,
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


def test_default_policy_remains_the_fallback(tmp_path):
    manager = AuthorityManager(
        policy=AuthorityPolicy(maximum_automatic_risk=RiskLevel.MEDIUM),
        storage_path=tmp_path / "authority.json",
    )

    authorization = manager.authorize(definition(risk="medium"))

    assert authorization.allowed is True
    assert authorization.requires_confirmation is False


def test_grant_can_override_default_high_risk_confirmation(tmp_path):
    manager = AuthorityManager(
        policy=AuthorityPolicy(maximum_automatic_risk=RiskLevel.MEDIUM),
        storage_path=tmp_path / "authority.json",
    )
    manager.grant("test_tool", reason="Trusted local capability.")

    authorization = manager.authorize(definition(risk="high"))

    assert authorization.allowed is True
    assert authorization.requires_confirmation is False
    assert manager.get_rule("test_tool").mode is AuthorityMode.AUTO


def test_rules_reload_across_manager_instances(tmp_path):
    path = tmp_path / "authority.json"

    first = AuthorityManager(storage_path=path)
    first.grant("test_tool", reason="Remember this decision.")
    first.require_confirmation("confirm_tool")
    first.deny("blocked_tool", reason="Blocked in this workspace.")

    second = AuthorityManager(storage_path=path)

    assert second.get_rule("test_tool").mode is AuthorityMode.AUTO
    assert second.get_rule("test_tool").reason == "Remember this decision."
    assert second.get_rule("confirm_tool").mode is AuthorityMode.CONFIRM
    assert second.get_rule("blocked_tool").mode is AuthorityMode.DENY


def test_confirm_rule_requires_confirmation_until_explicitly_confirmed(tmp_path):
    manager = AuthorityManager(storage_path=tmp_path / "authority.json")
    manager.require_confirmation("test_tool")

    blocked = manager.authorize(definition())
    confirmed = manager.authorize(definition(), confirmed=True)

    assert blocked.allowed is False
    assert blocked.requires_confirmation is True
    assert confirmed.allowed is True


def test_deny_rule_blocks_even_low_risk_capability(tmp_path):
    manager = AuthorityManager(storage_path=tmp_path / "authority.json")
    manager.deny("test_tool", reason="Blocked for this session.")

    authorization = manager.authorize(definition())

    assert authorization.allowed is False
    assert authorization.requires_confirmation is False
    assert "Blocked for this session." in authorization.reason


def test_clear_rule_persists_removal(tmp_path):
    path = tmp_path / "authority.json"

    manager = AuthorityManager(storage_path=path)
    manager.grant("test_tool")
    assert manager.clear_rule("test_tool") is True

    reloaded = AuthorityManager(storage_path=path)
    assert reloaded.get_rule("test_tool") is None


def test_store_uses_versioned_json(tmp_path):
    path = tmp_path / "authority.json"

    store = AuthorityStore(path)
    store.save((
        {"tool": "one", "mode": "auto", "reason": ""},
    ))

    payload = path.read_text(encoding="utf-8")
    assert '"schema_version": 1' in payload
    assert '"mode": "auto"' in payload


def test_kernel_accepts_explicit_authority_path(tmp_path):
    path = tmp_path / "authority.json"
    kernel = Kernel(authority_path=path)
    kernel.authority_manager.grant("echo")

    reloaded = Kernel(authority_path=path)
    assert reloaded.authority_manager.get_rule("echo").mode is AuthorityMode.AUTO


def test_kernel_dispatch_uses_persisted_authority_rule(tmp_path):
    path = tmp_path / "authority.json"

    writer = Kernel(authority_path=path)
    writer.register_tool(EchoTool())
    writer.authority_manager.deny("echo", reason="No echo capability.")

    reader = Kernel(authority_path=path)
    reader.register_tool(EchoTool())

    result = reader.tool_dispatcher.dispatch(
        ToolRequest(
            tool="echo",
            arguments={"text": "hello"},
            request_id="authority-test",
        )
    )

    assert result.success is False
    assert "No echo capability." in result.error
    assert result.metadata["requires_confirmation"] is False
