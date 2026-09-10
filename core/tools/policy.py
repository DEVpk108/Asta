from enum import IntEnum

from core.contracts import ToolDefinition


class RiskLevel(IntEnum):
    """Relative risk of a tool operation."""

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
        self.requires_confirmation = requires_confirmation


class AuthorityPolicy:
    """Central policy engine for tool authorization.

    By default, A.S.T.A. can perform low- and medium-risk everyday actions
    without interrupting the user. High- and critical-risk actions require
    explicit confirmation unless a caller supplies an elevated authority
    policy intentionally.
    """

    def __init__(
        self,
        maximum_automatic_risk: RiskLevel = RiskLevel.MEDIUM,
    ):
        self.maximum_automatic_risk = maximum_automatic_risk

    def authorize(
        self,
        definition: ToolDefinition,
        *,
        confirmed: bool = False,
    ) -> AuthorizationResult:
        risk = self._parse_risk(definition.risk_level)

        # Explicit confirmation is accepted only for a tool that was
        # identified as requiring confirmation. Low- and medium-risk tools
        # continue to follow the automatic policy and do not need a
        # confirmation path unless the policy says otherwise.
        if definition.requires_confirmation:
            if confirmed:
                return AuthorizationResult(
                    allowed=True,
                    reason="Tool execution explicitly confirmed.",
                )

            return AuthorizationResult(
                allowed=False,
                reason="User confirmation is required for this tool.",
                requires_confirmation=True,
            )

        if risk > self.maximum_automatic_risk:
            if confirmed:
                return AuthorizationResult(
                    allowed=True,
                    reason="Tool execution explicitly confirmed.",
                )

            return AuthorizationResult(
                allowed=False,
                reason=(
                    f"Tool risk level '{risk.name}' exceeds the automatic "
                    "execution policy."
                ),
                requires_confirmation=True,
            )

        return AuthorizationResult(
            allowed=True,
            reason="Tool authorized.",
        )

    @staticmethod
    def _parse_risk(value: str) -> RiskLevel:
        normalized = str(value).strip().upper()

        try:
            return RiskLevel[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unknown tool risk level: {value}"
            ) from exc
