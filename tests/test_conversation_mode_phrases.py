import pytest

from ai.ai_module import AIModule


class _Bus:
    def __init__(self):
        self.events = []

    def emit(self, name, **payload):
        self.events.append((name, payload))


def _module():
    module = AIModule.__new__(AIModule)
    module.event_bus = _Bus()
    module.spoken = []
    module._emit_assistant_text = module.spoken.append
    return module


@pytest.mark.parametrize(
    "heard",
    ["Then of conversation mode", "turn of conversation mode", "Okay, done off the conversation mode."],
)
def test_misheard_turn_off_conversation_mode_still_turns_it_off(heard):
    module = _module()

    assert module._handle_conversation_mode_command(heard) is True
    assert module.event_bus.events == [("conversation_mode_set", {"enabled": False})]


def test_unrelated_sentence_is_not_a_conversation_mode_command():
    module = _module()

    assert module._handle_conversation_mode_command("then open spotify") is False
    assert module.event_bus.events == []
