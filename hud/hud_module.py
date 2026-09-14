from core.module import Module

from chat_history import ChatHistoryStore

from .hud_state import HUDState
from .transport import HUDTransport


_UNSET = object()
_WAKEWORD_CHAT_SUPPRESSED = {
    "yes?",
    "hello! how can i help?",
}


class HUDModule(Module):
    VALID_MODES = {"idle", "listening", "thinking", "speaking", "executing", "approval", "error"}
    VALID_INTENSITIES = {"low", "medium", "high"}

    def __init__(self, kernel):
        super().__init__(name="HUD", event_bus=kernel.event_bus, kernel=kernel)
        self.state = HUDState()
        self.transport = HUDTransport()
        self.chat_history = ChatHistoryStore()
        self._conversation_active = False
        self._runtime_ready = False

    def initialize(self):
        self.event_bus.subscribe("voice_ready", self.on_voice_ready)
        self.event_bus.subscribe("assistant_sentence", self.on_assistant_sentence)
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.subscribe("speech_started", self.on_speech_started)
        self.event_bus.subscribe("speech_finished", self.on_speech_finished)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.subscribe("speech_audio_level", self.on_speech_audio_level)
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe("tool_confirmation_required", self.on_tool_confirmation_required)
        self.event_bus.subscribe("tool_confirmation_response", self.on_tool_confirmation_response)
        self.transport.set_command_handler(self.on_transport_message)
        try:
            self.chat_history.initialize()
            self.transport.start()
            self.transport.publish_lifecycle("starting")
            self._publish_chat_context()
            self._publish_state()
            self.transport.publish_audio_level(0.0)
            print(f"[Chat] History ready: {self.chat_history.db_path}", flush=True)
        except OSError as exc:
            print(f"[HUD] Transport unavailable: {type(exc).__name__}: {exc}", flush=True)

    def _publish_chat_context(self):
        sessions = self.chat_history.sessions()
        self.transport.publish_chat_history(
            self.chat_history.recent(),
            session_id=self.chat_history.session_id,
            sessions=sessions,
        )
        self.transport.publish_chat_sessions(sessions)

    def _publish_chat_index(self):
        self.transport.publish_chat_sessions(self.chat_history.sessions())

    def get_state(self):
        return self.state

    def set_state(self, *, mode=None, intensity=None, status=None, progress=_UNSET, activity=_UNSET):
        if mode is not None:
            mode = str(mode).lower()
            if mode not in self.VALID_MODES:
                raise ValueError(f"Unsupported HUD mode: {mode}")
            self.state.mode = mode
        if intensity is not None:
            intensity = str(intensity).lower()
            if intensity not in self.VALID_INTENSITIES:
                raise ValueError(f"Unsupported HUD intensity: {intensity}")
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
            self.state.activity = None if activity is None else str(activity)
        self._publish_state()
        return self.state

    def _publish_state(self):
        try:
            self.transport.publish_state(self.state)
        except OSError as exc:
            print(f"[HUD] State publish failed: {type(exc).__name__}: {exc}", flush=True)

    def reset_state(self):
        self.state = HUDState()
        self._conversation_active = False
        self._runtime_ready = False
        self._publish_state()
        self.transport.publish_audio_level(0.0)
        return self.state

    def on_voice_ready(self):
        if self._runtime_ready:
            return
        self._runtime_ready = True
        print("[HUD] Voice listener ready; publishing runtime ready.", flush=True)
        self.set_state(mode="idle", intensity="low", status="READY", progress=None, activity="runtime")
        self.transport.publish_lifecycle("ready")

    def on_conversation_mode_set(self, enabled):
        self._conversation_active = bool(enabled)
        if self._conversation_active:
            self.set_state(mode="listening", intensity="medium", status="LISTENING", progress=None, activity="conversation")
        else:
            self.set_state(mode="idle", intensity="low", status="IDLE", progress=None, activity=None)

    def on_transport_message(self, message):
        if not isinstance(message, dict):
            return
        message_type = message.get("type")
        if message_type == "hud.shutdown":
            print("[HUD] Shutdown requested by Electron HUD.", flush=True)
            try:
                self.transport.publish_lifecycle("stopping")
            except OSError:
                pass
            self.kernel.shutdown()
            return

        if message_type == "hud.chat_select":
            session_id = str(message.get("session_id") or "").strip()
            if self.chat_history.switch_session(session_id):
                print(f"[Chat] Selected session {session_id}", flush=True)
                sessions = self.chat_history.sessions()
                self.transport.publish_chat_history(
                    self.chat_history.recent(),
                    session_id=self.chat_history.session_id,
                    sessions=sessions,
                )
                self.transport.publish_chat_sessions(sessions)
            return

        if message_type == "hud.chat_new":
            session_id = self.chat_history.new_session()
            print(f"[Chat] New session {session_id}", flush=True)
            sessions = self.chat_history.sessions()
            self.transport.publish_chat_history([], session_id=session_id, sessions=sessions)
            self.transport.publish_chat_sessions(sessions)
            return

        if not self._runtime_ready:
            print("[HUD] Ignoring text input while A.S.T.A. is still booting.", flush=True)
            return
        if message_type != "hud.input":
            return
        payload = message.get("input") or {}
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            return
        text = text.strip()
        if not text:
            return
        self.event_bus.emit("user_message", text=text)

    def on_user_message(self, text):
        if isinstance(text, str) and text.strip():
            value = text.strip()
            self.chat_history.append("user", value)
            self.transport.publish_chat(role="user", text=value)
            self._publish_chat_index()
        if self.state.mode in {"speaking", "approval", "executing"}:
            return
        self.set_state(mode="thinking", intensity="high", status="THINKING", progress=None, activity="reasoning")

    def on_speech_started(self, *args, **kwargs):
        self.set_state(mode="speaking", intensity="high", status="SPEAKING", progress=None, activity="speech")
        self.transport.publish_audio_level(0.0)

    def on_speech_finished(self, *args, **kwargs):
        if self._conversation_active:
            self.set_state(mode="listening", intensity="medium", status="LISTENING", progress=None, activity="command")
        else:
            self.set_state(mode="idle", intensity="low", status="IDLE", progress=None, activity=None)
        self.transport.publish_audio_level(0.0)

    def on_speech_interrupt(self, *args, **kwargs):
        if self._conversation_active:
            self.set_state(mode="listening", intensity="medium", status="LISTENING", progress=None, activity="interrupt")
        else:
            self.set_state(mode="idle", intensity="low", status="IDLE", progress=None, activity=None)
        self.transport.publish_audio_level(0.0)

    def on_speech_audio_level(self, level=0.0, *args, **kwargs):
        try:
            value = float(level)
        except (TypeError, ValueError):
            value = 0.0
        self.transport.publish_audio_level(max(0.0, min(1.0, value)))

    def on_tool_request(self, request):
        self.set_state(mode="executing", intensity="high", status="EXECUTING", progress=None, activity=getattr(request, "tool", None) or "tool")

    def on_tool_confirmation_required(self, request=None, reason=None):
        self.set_state(mode="approval", intensity="high", status="APPROVAL", progress=None, activity=str(reason) if reason else "approval")

    def on_tool_confirmation_response(self, request_id=None, approved=False):
        if approved:
            self.set_state(mode="executing", intensity="high", status="EXECUTING", progress=None, activity="tool")
        elif self._conversation_active:
            self.set_state(mode="listening", intensity="medium", status="LISTENING", progress=None, activity="approval_rejected")
        else:
            self.set_state(mode="idle", intensity="low", status="IDLE", progress=None, activity=None)

    def on_assistant_sentence(self, text):
        if not isinstance(text, str) or not text.strip():
            return
        value = text.strip()
        if value.lower() in _WAKEWORD_CHAT_SUPPRESSED:
            print(f"[HUD] Suppressed wake-word acknowledgment from chat: {value!r}", flush=True)
            return
        self.chat_history.append("assistant", value)
        self.transport.publish_chat(role="assistant", text=value)
        self._publish_chat_index()

    def shutdown(self):
        for event_name, callback in (
            ("voice_ready", self.on_voice_ready),
            ("assistant_sentence", self.on_assistant_sentence),
            ("user_message", self.on_user_message),
            ("conversation_mode_set", self.on_conversation_mode_set),
            ("speech_started", self.on_speech_started),
            ("speech_finished", self.on_speech_finished),
            ("speech_interrupt", self.on_speech_interrupt),
            ("speech_audio_level", self.on_speech_audio_level),
            ("tool_request", self.on_tool_request),
            ("tool_confirmation_required", self.on_tool_confirmation_required),
            ("tool_confirmation_response", self.on_tool_confirmation_response),
        ):
            self.event_bus.unsubscribe(event_name, callback)
        self.transport.set_command_handler(None)
        try:
            self.transport.publish_audio_level(0.0)
        except OSError:
            pass
        self.transport.stop()
        self.chat_history.close()
