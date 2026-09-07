from uuid import uuid4

from core.contracts import IntentResult, IntentType, ToolRequest
from core.tools.selector import ToolSelector


class ToolRequestBuilder:
    """Convert structured intent into executable ToolRequests.

    The builder does not contain an action-to-tool map. It asks the registry-
    backed ToolSelector to discover the appropriate capability and then copies
    the intent entities into the tool contract.
    """

    def __init__(self, registry):
        self.selector = ToolSelector(registry)

    def build(self, intent: IntentResult) -> ToolRequest:
        if intent.intent != IntentType.COMMAND:
            raise ValueError(
                "ToolRequest can only be built from a command intent."
            )

        definition = self.selector.select(intent)

        arguments = dict(intent.entities)
        arguments.pop("action", None)

        return ToolRequest(
            tool=definition.name,
            arguments=arguments,
            request_id=str(uuid4()),
            timeout_seconds=definition.timeout_seconds,
            metadata={
                "intent": intent.intent.value,
                "intent_confidence": intent.confidence,
                "classifier": intent.classifier,
            },
        )
