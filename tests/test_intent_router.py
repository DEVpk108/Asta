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


def test_polite_open_command_is_actionable():
    result = IntentRouter().analyze("Could you please open Chrome?")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "open", "target": "chrome"}


def test_embedded_command_is_actionable():
    result = IntentRouter().analyze("Nothing else, open Chrome.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "open", "target": "chrome"}


def test_can_you_open_is_command_not_capability_question():
    result = IntentRouter().analyze("Can you open Spotify?")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "open", "target": "spotify"}
