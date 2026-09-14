from types import SimpleNamespace

from core.event_bus import EventBus
from hud.hud_module import HUDModule


class DummyRequest:
    tool = "run_command"


def build_hud():
    kernel = SimpleNamespace(event_bus=EventBus())
    hud = HUDModule(kernel)
    hud.initialize()
    return kernel, hud


def test_speech_lifecycle_updates_hud_state():
    kernel, hud = build_hud()

    kernel.event_bus.emit("speech_started")
    assert hud.get_state().mode == "speaking"
    assert hud.get_state().intensity == "high"
    assert hud.get_state().activity == "speech"

    kernel.event_bus.emit("speech_finished")
    assert hud.get_state().mode == "listening"
    assert hud.get_state().intensity == "medium"
    assert hud.get_state().activity == "command"

    hud.shutdown()


def test_conversation_mode_controls_hud_state():
    kernel, hud = build_hud()

    kernel.event_bus.emit("conversation_mode_set", enabled=True)
    assert hud.get_state().mode == "listening"
    assert hud.get_state().activity == "conversation"

    kernel.event_bus.emit("conversation_mode_set", enabled=False)
    assert hud.get_state().mode == "idle"
    assert hud.get_state().activity is None

    hud.shutdown()


def test_tool_and_approval_events_update_hud_state():
    kernel, hud = build_hud()

    kernel.event_bus.emit("tool_request", request=DummyRequest())
    assert hud.get_state().mode == "executing"
    assert hud.get_state().activity == "run_command"

    kernel.event_bus.emit(
        "tool_confirmation_required",
        request=DummyRequest(),
        reason="high risk",
    )
    assert hud.get_state().mode == "approval"
    assert hud.get_state().activity == "high risk"

    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id="test-request",
        approved=True,
    )
    assert hud.get_state().mode == "executing"

    kernel.event_bus.emit(
        "tool_confirmation_response",
        request_id="test-request",
        approved=False,
    )
    assert hud.get_state().mode == "listening"

    hud.shutdown()


def test_user_message_enters_thinking_without_overwriting_speech():
    kernel, hud = build_hud()

    kernel.event_bus.emit("user_message", text="hello")
    assert hud.get_state().mode == "thinking"
    assert hud.get_state().activity == "reasoning"

    kernel.event_bus.emit("speech_started")
    kernel.event_bus.emit("user_message", text="do not overwrite speaking")
    assert hud.get_state().mode == "speaking"

    hud.shutdown()
