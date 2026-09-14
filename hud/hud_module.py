from core.module import Module

from .hud_state import HUDState


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

    def get_state(self):
        """Return the current HUD presentation state."""
        return self.state

    def set_state(
        self,
        *,
        mode=None,
        intensity=None,
        status=None,
        progress=None,
        activity=None,
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

        if progress is not None:
            progress = float(progress)
            if not 0.0 <= progress <= 1.0:
                raise ValueError("HUD progress must be between 0.0 and 1.0")
            self.state.progress = progress

        if activity is not None:
            self.state.activity = str(activity)

        return self.state

    def reset_state(self):
        """Reset the HUD to its initial idle state."""
        self.state = HUDState()
        return self.state

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
