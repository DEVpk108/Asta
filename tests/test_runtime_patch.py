from ai.runtime_patch import _extract_post_response_screenshot, _is_screenshot_capture_request, _is_screenshot_open_request
from core.intent_router import IntentRouter


def test_natural_screenshot_commands_are_detected():
    assert _is_screenshot_capture_request("take a screenshot") is True
    assert _is_screenshot_capture_request("Okay, take a screenshot") is True
    assert _is_screenshot_capture_request("Okay, I'll start take a screenshot") is True
    assert _is_screenshot_capture_request("capture the screen shot") is True


def test_negative_screenshot_command_is_not_executed():
    assert _is_screenshot_capture_request("don't take a screenshot") is False
    assert _is_screenshot_capture_request("do not capture a screenshot") is False


def test_open_screenshot_variants_are_detected():
    assert _is_screenshot_open_request("open screenshot") is True
    assert _is_screenshot_open_request("open screen shot") is True
    assert _is_screenshot_open_request("open the screen shot.") is True
    assert _is_screenshot_open_request("show screen shot") is True


def test_compound_screenshot_command_is_claimed_by_intent_router():
    result = IntentRouter().analyze(
        "open Chrome, then open camera, then take a screenshot"
    )

    assert result.intent.value == "command"
    assert len(result.entities["commands"]) == 3
    assert result.entities["commands"][0]["action"] == "open"
    assert result.entities["commands"][0]["target"] == "chrome"
    assert result.entities["commands"][1]["action"] == "open"
    assert result.entities["commands"][1]["target"] == "camera"
    assert result.entities["commands"][2]["action"] == "screenshot"


def test_mixed_request_still_extracts_conversational_prefix():
    assert (
        _extract_post_response_screenshot("tell me a joke then take a screenshot")
        == "tell me a joke"
    )
    assert (
        _extract_post_response_screenshot("tell me about yourself and then capture a screenshot")
        == "tell me about yourself"
    )
