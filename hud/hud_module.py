from core.module import Module

from .hud_state import HUDState


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

    def initialize(self):
        self.event_bus.subscribe(
            "assistant_response",
            self.on_assistant_response,
        )
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("speech_started", self.on_speech_started)
        self.event_bus.subscribe("speech_finished", self.on_speech_finished)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self.on_tool_confirmation_response,
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

        return self.state

    def reset_state(self):
        """Reset the HUD to its initial idle state."""
        self.state = HUDState()
        return self.state

    def on_user_message(self, text):
        """Enter thinking state after a user command reaches the AI layer.

        The EventBus is synchronous, so the AI and speech modules may already
        have advanced the HUD to a newer state by the time this callback runs.
        Do not overwrite speaking/approval states that are already active.
        """
        if self.state.mode not in {"speaking", "approval"}:
            self.set_state(
                mode="thinking",
                intensity="high",
                status="THINKING",
                progress=None,
                activity="reasoning",
            )

    def on_speech_started(self, *args, **kwargs):
        self.set_state(
            mode="speaking",
            intensity="high",
            status="SPEAKING",
            progress=None,
            activity="speech",
        )

    def on_speech_finished(self, *args, **kwargs):
        self.set_state(
            mode="idle",
            intensity="low",
            status="IDLE",
            progress=None,
            activity=None,
        )

    def on_speech_interrupt(self, *args, **kwargs):
        self.set_state(
            mode="listening",
            intensity="medium",
            status="LISTENING",
            progress=None,
            activity="interrupt",
        )

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
        # Let runtime patches and tools wait until the HUD has handed the
        # completed response to the terminal before taking a desktop screenshot.
        self.event_bus.emit("hud_rendered", text=text)

    def shutdown(self):
        self.event_bus.unsubscribe(
            "assistant_response",
            self.on_assistant_response,
        )
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("speech_started", self.on_speech_started)
        self.event_bus.unsubscribe("speech_finished", self.on_speech_finished)
        self.event_bus.unsubscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_tool_confirmation_response,
        )
