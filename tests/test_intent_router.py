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


def test_screen_shot_is_actionable():
    result = IntentRouter().analyze("Take a screen shot!")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "screenshot"}


def test_take_the_screen_shot_is_actionable():
    result = IntentRouter().analyze("Take the screen shot!")
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


def test_compound_command_with_and_is_sequenced():
    result = IntentRouter().analyze("Open WhatsApp and take screen shot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "whatsapp"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_then_is_sequenced():
    result = IntentRouter().analyze("Open Chrome then take the screen shot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "chrome"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_comma_then_is_sequenced():
    result = IntentRouter().analyze("Open WhatsApp, then take screenshot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "whatsapp"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_and_then_is_sequenced():
    result = IntentRouter().analyze("Open Chrome and then take screenshot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "chrome"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_comma_after_first_action():
    result = IntentRouter().analyze("Open Chrome, and then take screenshot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "chrome"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_comma_before_close():
    result = IntentRouter().analyze("Close camera, close Chrome.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "close", "target": "camera"},
            {"action": "close", "target": "chrome"},
        ]
    }


def test_compound_command_with_implicit_screenshot_suffix():
    result = IntentRouter().analyze("Open camera take screenshot.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "camera"},
            {"action": "screenshot"},
        ]
    }


def test_compound_command_with_multiple_natural_steps():
    result = IntentRouter().analyze(
        "Open Chrome then open camera take screen shot"
    )
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "commands": [
            {"action": "open", "target": "chrome"},
            {"action": "open", "target": "camera"},
            {"action": "screenshot"},
        ]
    }
