from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .policy import AuthorizationResult, AuthorityPolicy
from ..contracts import ToolDefinition


class AuthorityMode(StrEnum):
    """Explicit per-capability authority overrides."""

    AUTO = "auto"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class AuthorityRule:
    """One explicit authority rule for a capability."""

    tool: str
    mode: AuthorityMode
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "mode": self.mode.value,
            "reason": self.reason,
        }


class AuthorityManager:
    """Stateful authority layer built on top of the default risk policy.

    The manager provides explicit per-tool overrides while preserving the
    existing AuthorityPolicy as the safe default for anything not explicitly
    configured. Rules are intentionally runtime state for now; a later
    persistence layer can serialize snapshot() without changing dispatch.
    """

    def __init__(self, *, policy: AuthorityPolicy | None = None, event_bus=None):
        self.policy = policy if policy is not None else AuthorityPolicy()
        self.event_bus = event_bus
        self._rules: dict[str, AuthorityRule] = {}

    def set_rule(
        self,
        tool: str,
        mode: AuthorityMode | str,
        *,
        reason: str = "",
    ) -> AuthorityRule:
        name = str(tool).strip()
        if not name:
            raise ValueError("tool name must be non-empty")

        try:
            normalized = mode if isinstance(mode, AuthorityMode) else AuthorityMode(str(mode).strip().lower())
        except ValueError as exc:
            raise ValueError(f"unknown authority mode: {mode}") from exc

        rule = AuthorityRule(
            tool=name,
            mode=normalized,
            reason=str(reason).strip(),
        )
        self._rules[name] = rule

        if self.event_bus is not None:
            self.event_bus.emit(
                "authority_rule_updated",
                rule=rule.to_dict(),
            )

        return rule

    def grant(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.AUTO, reason=reason)

    def require_confirmation(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.CONFIRM, reason=reason)

    def deny(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.DENY, reason=reason)

    def clear_rule(self, tool: str) -> bool:
        name = str(tool).strip()
        removed = self._rules.pop(name, None) is not None

        if removed and self.event_bus is not None:
            self.event_bus.emit(
                "authority_rule_cleared",
                tool=name,
            )

        return removed

    def get_rule(self, tool: str) -> AuthorityRule | None:
        return self._rules.get(str(tool).strip())

    def list_rules(self) -> tuple[AuthorityRule, ...]:
        return tuple(self._rules.values())

    def snapshot(self) -> tuple[dict[str, str], ...]:
        return tuple(rule.to_dict() for rule in self._rules.values())

    def authorize(
        self,
        definition: ToolDefinition,
        *,
        confirmed: bool = False,
    ) -> AuthorizationResult:
        rule = self.get_rule(definition.name)

        if rule is None:
            return self.policy.authorize(
                definition,
                confirmed=confirmed,
            )

        if rule.mode is AuthorityMode.DENY:
            return AuthorizationResult(
                allowed=False,
                reason=rule.reason or "Capability is blocked by the authority policy.",
                requires_confirmation=False,
            )

        if rule.mode is AuthorityMode.AUTO:
            return AuthorizationResult(
                allowed=True,
                reason=rule.reason or "Capability explicitly trusted by the authority manager.",
            )

        if confirmed:
            return AuthorizationResult(
                allowed=True,
                reason="Capability execution explicitly confirmed.",
            )

        return AuthorizationResult(
            allowed=False,
            reason=rule.reason or "User confirmation is required for this capability.",
            requires_confirmation=True,
        )
