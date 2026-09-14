from core.module import Module

from .hud_state import HUDState
from .transport import HUDTransport


_UNSET = object()


class HUDModule(Module):

    VALID_MODES = {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "executing",
        "approval",
        "error",
    }

    VALID_INTENSITIES = {
        "low",
        "medium",
        "high",
    }

    def __init__(self, kernel):
        super().__init__(
            name="HUD",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.state = HUDState()
        self.transport = HUDTransport()

    def initialize(self):
        self.event_bus.subscribe(
            "assistant_response",
            self.on_assistant_response,
        )
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.subscribe("speech_started", self.on_speech_started)
        self.event_bus.subscribe("speech_finished", self.on_speech_finished)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.subscribe("speech_audio_level", self.on_speech_audio_level)
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self.on_tool_confirmation_response,
        )

        try:
            self.transport.start()
            self._publish_state()
            self.transport.publish_audio_level(0.0)
        except OSError as exc:
            # The HUD is optional infrastructure. A transport bind failure
            # must not prevent the rest of A.S.T.A. from starting.
            print(
                f"[HUD] Transport unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def get_state(self):
        """Return the current HUD presentation state."""
        return self.state

    def set_state(
        self,
        *,
        mode=None,
        intensity=None,
        status=None,
        progress=_UNSET,
        activity=_UNSET,
    ):
        """Update the HUD presentation state with validated values."""
        if mode is not None:
            mode = str(mode).lower()
            if mode not in self.VALID_MODES:
                raise ValueError(f"Unsupported HUD mode: {mode}")
            self.state.mode = mode

        if intensity is not None:
            intensity = str(intensity).lower()
            if intensity not in self.VALID_INTENSITIES:
                raise ValueError(
                    f"Unsupported HUD intensity: {intensity}"
                )
            self.state.intensity = intensity

        if status is not None:
            self.state.status = str(status)

        if progress is not _UNSET:
            if progress is None:
                self.state.progress = None
            else:
                progress = float(progress)
                if not 0.0 <= progress <= 1.0:
                    raise ValueError("HUD progress must be between 0.0 and 1.0")
                self.state.progress = progress

        if activity is not _UNSET:
            self.state.activity = (
                None if activity is None else str(activity)
            )

        self._publish_state()
        return self.state

    def _publish_state(self):
        try:
            self.transport.publish_state(self.state)
        except OSError as exc:
            print(
                f"[HUD] State publish failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def reset_state(self):
        """Reset the HUD to its initial idle state."""
        self.state = HUDState()
        self._publish_state()
        self.transport.publish_audio_level(0.0)
        return self.state

    def on_user_message(self, text):
        """Enter thinking state as soon as the user command reaches the kernel.

        HUD is registered before AIModule so this callback runs before model
        inference. Synchronous EventBus dispatch therefore gives the renderer a
        real THINKING transition instead of applying it after inference.
        """
        if self.state.mode in {"speaking", "approval", "executing"}:
            return
        self.set_state(
            mode="thinking",
            intensity="high",
            status="THINKING",
            progress=None,
            activity="reasoning",
        )

    def on_conversation_mode_set(self, enabled):
        if enabled:
            self.set_state(
                mode="listening",
                intensity="medium",
                status="LISTENING",
                progress=None,
                activity="conversation",
            )
        else:
            self.set_state(
                mode="idle",
                intensity="low",
                status="IDLE",
                progress=None,
                activity=None,
            )

    def on_speech_started(self, *args, **kwargs):
        self.set_state(
            mode="speaking",
            intensity="high",
            status="SPEAKING",
            progress=None,
            activity="speech",
        )
        self.transport.publish_audio_level(0.0)

    def on_speech_finished(self, *args, **kwargs):
        # A.S.T.A. is voice-first, so after an utterance completes the natural
        # next presentation state is waiting for the user's next command.
        self.set_state(
            mode="listening",
            intensity="medium",
            status="LISTENING",
            progress=None,
            activity="command",
        )
        self.transport.publish_audio_level(0.0)

    def on_speech_interrupt(self, *args, **kwargs):
        self.set_state(
            mode="listening",
            intensity="medium",
            status="LISTENING",
            progress=None,
            activity="interrupt",
        )
        self.transport.publish_audio_level(0.0)

    def on_speech_audio_level(self, level=0.0, *args, **kwargs):
        try:
            value = float(level)
        except (TypeError, ValueError):
            value = 0.0
        self.transport.publish_audio_level(max(0.0, min(1.0, value)))

    def on_tool_request(self, request):
        tool_name = getattr(request, "tool", None)
        self.set_state(
            mode="executing",
            intensity="high",
            status="EXECUTING",
            progress=None,
            activity=tool_name or "tool",
        )

    def on_tool_confirmation_required(self, request=None, reason=None):
        activity = "approval"
        if reason:
            activity = str(reason)
        self.set_state(
            mode="approval",
            intensity="high",
            status="APPROVAL",
            progress=None,
            activity=activity,
        )

    def on_tool_confirmation_response(self, request_id=None, approved=False):
        if approved:
            self.set_state(
                mode="executing",
                intensity="high",
                status="EXECUTING",
                progress=None,
                activity="tool",
            )
        else:
            self.set_state(
                mode="listening",
                intensity="medium",
                status="LISTENING",
                progress=None,
                activity="approval_rejected",
            )

    def on_assistant_response(self, text):
        print(
            f"[HUD] {text}",
            flush=True,
        )
        self.event_bus.emit("hud_rendered", text=text)

    def shutdown(self):
        self.event_bus.unsubscribe(
            "assistant_response",
            self.on_assistant_response,
        )
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.unsubscribe("speech_started", self.on_speech_started)
        self.event_bus.unsubscribe("speech_finished", self.on_speech_finished)
        self.event_bus.unsubscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.unsubscribe("speech_audio_level", self.on_speech_audio_level)
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_tool_confirmation_response,
        )
        try:
            self.transport.publish_audio_level(0.0)
        except OSError:
            pass
        self.transport.stop()
