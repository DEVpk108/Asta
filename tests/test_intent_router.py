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


def test_create_note_command_is_actionable():
    result = IntentRouter().analyze(
        "Take a note: test the new barge in detector."
    )
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "create_note",
        "content": "test the new barge in detector",
    }


def test_titled_note_command_extracts_title_and_content():
    result = IntentRouter().analyze(
        "Write a note titled Voice Test saying check interruption."
    )
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "create_note",
        "title": "voice test",
        "content": "check interruption",
    }


def test_read_note_command_is_actionable():
    result = IntentRouter().analyze("Read note Voice Test.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "read_note",
        "target": "voice test",
    }


def test_list_notes_command_is_actionable():
    result = IntentRouter().analyze("Show my notes.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "list_notes"}


def test_search_notes_command_is_actionable():
    result = IntentRouter().analyze("Search notes for interruption.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "search_notes",
        "query": "interruption",
    }


def test_collapsed_open_command_is_actionable():
    result = IntentRouter().analyze("OpenSpotify")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "open", "target": "spotify"}


def test_collapsed_close_pronoun_is_actionable():
    result = IntentRouter().analyze("Closeit")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {"action": "close", "target": "it"}


def test_collapsed_open_command_supports_multiword_app_without_spaces():
    result = IntentRouter().analyze("OpenVisualStudioCode")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "open",
        "target": "visualstudiocode",
    }



def test_media_play_command_is_actionable():
    result = IntentRouter().analyze("Play Hanuman Chalisa on Spotify")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "media",
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }


def test_media_control_command_is_actionable():
    result = IntentRouter().analyze("Pause the music.")
    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "media",
        "operation": "pause",
    }



def test_intent_router_recovers_media_command_when_stt_drops_play():
    router = IntentRouter(media_providers=("spotify", "system"))

    result = router.analyze("Only Hanuman Chalisa on Spotify")

    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "media",
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }


def test_intent_router_strips_consumed_wakeword_from_command():
    router = IntentRouter(media_providers=("spotify", "system"))

    result = router.analyze("Hey Asta, play Hanuman Chalisa on Spotify")

    assert result.intent == IntentType.COMMAND
    assert result.normalized_text == "play hanuman chalisa on spotify"
    assert result.entities == {
        "action": "media",
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }
