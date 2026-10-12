"""Model-written tool calls are routed safely and never spoken."""
from types import SimpleNamespace

from ai.model_tool_calls import (
    finish_model_response,
    media_entities,
    native_tool_call_text,
    speech_filter,
    split_tool_calls,
)
from core.intent_router import IntentRouter

PAUSE = "<|tool_call_start|>[media.control(action='pause')]<|tool_call_end|>"


class FakeBus:
    def __init__(self):
        self.events = []

    def emit(self, name, **kwargs):
        self.events.append((name, kwargs))


def make_module():
    routed = []
    history = SimpleNamespace(history=[{"role": "user", "content": "Pause the so"}, {"role": "assistant", "content": PAUSE}])
    module = SimpleNamespace(event_bus=FakeBus(), engine=history, _handle_command_intent=routed.append)
    return module, routed, history


def test_lfm_pause_call_is_parsed():
    clean, calls = split_tool_calls(PAUSE)
    assert clean == ""
    assert media_entities(calls) == {"action": "media", "operation": "pause"}


def test_pause_call_is_routed_and_not_shown():
    module, routed, history = make_module()
    assert finish_model_response(module, PAUSE) is None
    assert routed and routed[0].entities == {"action": "media", "operation": "pause"}
    assert "tool_call" not in history.history[-1]["content"]
    assert not module.event_bus.events


def test_unlisted_tool_call_is_dropped_with_spoken_fallback():
    module, routed, _ = make_module()
    reply = finish_model_response(module, "<|tool_call_start|>[system.run_command(command='del x')]<|tool_call_end|>")
    assert not routed
    assert reply and "tool" not in reply.lower()
    assert module.event_bus.events[0][0] == "assistant_sentence"


def test_plain_reply_is_unchanged():
    module, routed, _ = make_module()
    assert finish_model_response(module, "Hello! How can I help?") == "Hello! How can I help?"
    assert not routed


def test_markup_is_never_spoken():
    spoken = []
    say = speech_filter(spoken.append)
    say(PAUSE)
    say("[media.control(action='pause')]")
    say("Sure. " + PAUSE)
    say("All done.")
    assert spoken == ["Sure.", "All done."]


def test_streamed_native_tool_calls_render_as_markup():
    text = native_tool_call_text([{"function": {"name": "media.control", "arguments": '{"operation": "next"}'}}])
    assert media_entities(split_tool_calls(text)[1]) == {"action": "media", "operation": "next"}


def test_clipped_pause_commands_are_understood():
    router = IntentRouter()
    for text in ("Pause the so", "pause the son", "pause the mus", "pause the song"):
        result = router.analyze(text)
        assert result.intent.value == "command", text
        assert result.entities == {"action": "media", "operation": "pause"}, text
    assert router.analyze("pause so much").intent.value != "command"
