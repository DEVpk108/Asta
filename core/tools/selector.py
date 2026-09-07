from core.contracts import IntentResult, IntentType, ToolDefinition


class ToolSelector:
    """Select the best registered tool for a structured command intent.

    IntentRouter owns language understanding. ToolSelector owns capability
    discovery and deterministic selection. ToolDispatcher owns execution and
    AuthorityPolicy owns authorization.
    """

    def __init__(self, registry):
        self.registry = registry

    def select(self, intent: IntentResult) -> ToolDefinition:
        if intent.intent != IntentType.COMMAND:
            raise ValueError(
                "Only command intents can select executable tools."
            )

        action = self._normalize_action(intent.entities.get("action"))
        if not action:
            raise ValueError(
                "Command intent is missing a valid action."
            )

        candidates = []

        for definition in self.registry.definitions():
            score = self._score(definition, action, intent.entities)
            if score > 0:
                candidates.append((score, definition.name, definition))

        if not candidates:
            available = ", ".join(self.registry.list()) or "none"
            raise ValueError(
                f"No registered tool supports command action '{action}'. "
                f"Available tools: {available}"
            )

        # Highest capability score wins. Name is a deterministic final
        # tie-breaker so selection never depends on dictionary ordering.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    @staticmethod
    def _normalize_action(value) -> str | None:
        if not isinstance(value, str):
            return None

        value = value.strip().lower()
        return value or None

    @classmethod
    def _score(
        cls,
        definition: ToolDefinition,
        action: str,
        entities: dict,
    ) -> int:
        metadata = definition.metadata or {}

        actions = cls._normalized_values(metadata.get("actions"))
        aliases = cls._normalized_values(metadata.get("action_aliases"))

        if action in actions:
            score = 100
        elif action in aliases:
            score = 90
        else:
            return 0

        schema = definition.input_schema or {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if not isinstance(properties, dict):
            properties = {}
        if not isinstance(required, (list, tuple, set, frozenset)):
            required = ()

        # Prefer capabilities whose schemas actually consume the entities
        # produced by the intent layer.
        for key, value in entities.items():
            if key == "action" or not value:
                continue
            if key in properties:
                score += 5

        # A tool that requires an entity missing from the intent is not a
        # viable candidate, even if it advertises the requested action.
        if any(
            key != "action" and key not in entities
            for key in required
        ):
            return 0

        # Exact action declarations are preferred over aliases; required
        # inputs are a useful secondary signal when multiple tools overlap.
        score += 2 * sum(
            1
            for key in required
            if key != "action" and key in entities
        )

        return score

    @staticmethod
    def _normalized_values(value) -> set[str]:
        if not isinstance(value, (list, tuple, set, frozenset)):
            return set()

        return {
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        }
