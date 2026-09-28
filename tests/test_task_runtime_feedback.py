from core.contracts.intent import IntentResult, IntentType
from core.task_runtime import TaskRuntimeModule


def make_intent(entities):
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="test",
        entities=entities,
        requires_tools=True,
        classifier="rules",
    )


def test_media_command_acknowledges_before_execution():
    intent = make_intent(
        {
            "action": "media",
            "operation": "play",
            "query": "Hanuman Chalisa",
            "provider": "spotify",
        }
    )

    assert TaskRuntimeModule._acknowledgment_for_intent(intent) == (
        "Okay, sir. Playing Hanuman Chalisa on spotify."
    )


def test_open_command_acknowledgment():
    intent = make_intent(
        {
            "action": "open",
            "target": "spotify",
        }
    )

    assert TaskRuntimeModule._acknowledgment_for_intent(intent) == (
        "Okay, sir. Opening spotify."
    )
