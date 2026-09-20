from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .policy import AuthorizationResult, AuthorityPolicy
from .authority_store import AuthorityStore
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

    Explicit rules are persisted through AuthorityStore when configured.
    Anything without a rule continues to use AuthorityPolicy.
    """

    def __init__(
        self,
        *,
        policy: AuthorityPolicy | None = None,
        event_bus=None,
        storage_path=None,
        store: AuthorityStore | None = None,
    ):
        self.policy = policy if policy is not None else AuthorityPolicy()
        self.event_bus = event_bus
        self.store = store if store is not None else AuthorityStore(storage_path)
        self._rules: dict[str, AuthorityRule] = {}
        self._load_persisted_rules()

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
            normalized = (
                mode
                if isinstance(mode, AuthorityMode)
                else AuthorityMode(str(mode).strip().lower())
            )
        except ValueError as exc:
            raise ValueError(f"unknown authority mode: {mode}") from exc

        rule = AuthorityRule(
            tool=name,
            mode=normalized,
            reason=str(reason).strip(),
        )

        candidate = dict(self._rules)
        candidate[name] = rule
        self._persist(candidate)
        self._rules = candidate

        self._emit("authority_rule_updated", rule=rule.to_dict())
        return rule

    def grant(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.AUTO, reason=reason)

    def require_confirmation(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.CONFIRM, reason=reason)

    def deny(self, tool: str, *, reason: str = "") -> AuthorityRule:
        return self.set_rule(tool, AuthorityMode.DENY, reason=reason)

    def clear_rule(self, tool: str) -> bool:
        name = str(tool).strip()
        if name not in self._rules:
            return False

        candidate = dict(self._rules)
        del candidate[name]
        self._persist(candidate)
        self._rules = candidate

        self._emit("authority_rule_cleared", tool=name)
        return True

    def get_rule(self, tool: str) -> AuthorityRule | None:
        return self._rules.get(str(tool).strip())

    def list_rules(self) -> tuple[AuthorityRule, ...]:
        return tuple(self._rules.values())

    def snapshot(self) -> tuple[dict[str, str], ...]:
        return tuple(rule.to_dict() for rule in self._rules.values())

    def reload(self) -> tuple[AuthorityRule, ...]:
        self._rules.clear()
        self._load_persisted_rules()
        self._emit("authority_reloaded", rules=list(self.snapshot()))
        return self.list_rules()

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

    def _load_persisted_rules(self) -> None:
        for payload in self.store.load():
            try:
                rule = AuthorityRule(
                    tool=str(payload["tool"]).strip(),
                    mode=AuthorityMode(str(payload["mode"]).strip().lower()),
                    reason=str(payload.get("reason", "")).strip(),
                )
            except (KeyError, ValueError):
                continue

            if rule.tool:
                self._rules[rule.tool] = rule

    def _persist(self, rules: dict[str, AuthorityRule]) -> None:
        try:
            self.store.save(
                tuple(rule.to_dict() for rule in rules.values())
            )
        except OSError as exc:
            self._emit(
                "authority_persistence_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(
                f"Failed to persist authority rules: {exc}"
            ) from exc

    def _emit(self, event: str, **payload) -> None:
        if self.event_bus is not None:
            self.event_bus.emit(event, **payload)
