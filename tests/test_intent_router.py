from core.intent_router import IntentRouter
from core.contracts import IntentType


def test_screenshot_with_terminal_punctuation_is_actionable():
    result = IntentRouter().analyze("Screenshot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "screenshot"}


def test_take_screenshot_is_actionable():
    result = IntentRouter().analyze("Take a screenshot!")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "screenshot"}


def test_open_command_preserves_target_after_normalization():
    result = IntentRouter().analyze("Open Calculator.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "open", "target": "calculator"}
