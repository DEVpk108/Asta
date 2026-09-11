from ai.ai_module import AIModule
from voice.voice_module import VoiceModule


class DummyEventBus:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))


def test_conversation_mode_commands_are_recognized():
    assert AIModule._normalize_question("Conversation mode on!") == "conversation mode on"
    assert AIModule._normalize_question("Turn OFF conversation mode.") == "turn off conversation mode"
    assert "conversation mode on" in AIModule._CONVERSATION_MODE_ON
    assert "turn off conversation mode" in AIModule._CONVERSATION_MODE_OFF


def test_conversation_mode_on_emits_enable_event():
    ai = object.__new__(AIModule)
    ai.event_bus = DummyEventBus()

    handled = ai._handle_conversation_mode_command("conversation mode on")

    assert handled is True
    assert ai.event_bus.events[0] == (
        "conversation_mode_set",
        {"enabled": True},
    )
    assert ai.event_bus.events[1] == (
        "assistant_sentence",
        {"text": "Conversation mode is on. You can talk to me without the wake word."},
    )


def test_conversation_mode_off_emits_disable_event():
    ai = object.__new__(AIModule)
    ai.event_bus = DummyEventBus()

    handled = ai._handle_conversation_mode_command("turn off conversation mode")

    assert handled is True
    assert ai.event_bus.events[0] == (
        "conversation_mode_set",
        {"enabled": False},
    )


def test_voice_manual_mode_does_not_expire():
    voice = object.__new__(VoiceModule)
    voice._conversation_active = False
    voice._manual_conversation = False
    voice._last_interaction = 0.0
    voice.conversation_timeout = 0.0

    voice.on_conversation_mode_set(True)

    assert voice._conversation_active is True
    assert voice._manual_conversation is True
    assert voice._conversation_expired() is False

    voice.on_conversation_mode_set(False)
    assert voice._conversation_active is False
    assert voice._manual_conversation is False
