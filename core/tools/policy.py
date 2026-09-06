from enum import IntEnum

from core.contracts import ToolDefinition


class RiskLevel(IntEnum):
    """
    Relative risk of a tool operation.

    Higher values require stricter authorization.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AuthorizationResult:

    __slots__ = (
        "allowed",
        "reason",
        "requires_confirmation",
    )

    def __init__(
        self,
        *,
        allowed: bool,
        reason: str = "",
        requires_confirmation: bool = False,
    ):
        self.allowed = allowed
        self.reason = reason
        self.requires_confirmation = (
            requires_confirmation
        )


class AuthorityPolicy:
    """
    Central policy engine for tool authorization.

    This is intentionally independent from the dispatcher.
    """

    def __init__(
        self,
        maximum_automatic_risk: RiskLevel = (
            RiskLevel.LOW
        ),
    ):
        self.maximum_automatic_risk = (
            maximum_automatic_risk
        )

    def authorize(
        self,
        definition: ToolDefinition,
    ) -> AuthorizationResult:

        risk = self._parse_risk(
            definition.risk_level
        )

        # Explicit confirmation always wins.
        if definition.requires_confirmation:
            return AuthorizationResult(
                allowed=False,
                reason=(
                    "User confirmation is required "
                    "for this tool."
                ),
                requires_confirmation=True,
            )

        if risk > self.maximum_automatic_risk:
            return AuthorizationResult(
                allowed=False,
                reason=(
                    f"Tool risk level '{risk.name}' "
                    "exceeds the automatic execution "
                    "policy."
                ),
                requires_confirmation=True,
            )

        return AuthorizationResult(
            allowed=True,
            reason="Tool authorized.",
        )

    @staticmethod
    def _parse_risk(
        value: str,
    ) -> RiskLevel:

        normalized = (
            str(value)
            .strip()
            .upper()
        )

        try:
            return RiskLevel[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unknown tool risk level: {value}"
            ) from exc