from core.contracts import IntentResult, IntentType, ToolDefinition


class ToolSelector:
    """Select an executable tool from the registry using structured intent.

    Selection is deliberately definition-driven. IntentRouter decides what the
    user is trying to do; ToolSelector decides which registered capability is
    the best match. Tool execution and authority checks remain outside this
    class.
    """

    def __init__(self, registry):
        self.registry = registry

    def select(self, intent: IntentResult) -> ToolDefinition:
        if intent.intent != IntentType.COMMAND:
            raise ValueError("Only command intents can select executable tools.")

        action = intent.entities.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Command intent is missing a valid action.")

        action = action.strip().lower()
        candidates = []

        for definition in self.registry.definitions():
            metadata = definition.metadata or {}
            actions = metadata.get("actions", ())
            aliases = metadata.get("action_aliases", ())

            if not isinstance(actions, (list, tuple, set, frozenset)):
                actions = ()
            if not isinstance(aliases, (list, tuple, set, frozenset)):
                aliases = ()

            normalized_actions = {str(value).strip().lower() for value in actions}
            normalized_aliases = {str(value).strip().lower() for value in aliases}

            if action in normalized_actions:
                score = 100
            elif action in normalized_aliases:
                score = 90
            else:
                continue

            # Prefer tools whose description/schema can consume the entities
            # produced by IntentRouter. This keeps selection deterministic while
            # still allowing multiple tools to advertise the same action.
            properties = definition.input_schema.get("properties", {})
            entity_matches = sum(
                1 for key in intent.entities if key != "action" and key in properties
            )
            score += entity_matches

            candidates.append((score, definition.name, definition))

        if not candidates:
            raise ValueError(
                f"No registered tool supports command action: {action}"
            )

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]
