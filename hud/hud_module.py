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
        self.event_bus.subscribe("command_acknowledged", self.on_command_acknowledged)
        self.event_bus.subscribe("task_progress", self.on_task_progress)
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe(
            "incremental_user_message",
            self.on_incremental_user_message,
        )
        self.event_bus.subscribe(
            "incremental_voice_listening",
            self.on_incremental_voice_listening,
        )
        self.event_bus.subscribe(
            "voice_action_committed",
            self.on_voice_action_committed,
        )
        self.event_bus.subscribe("conversation_mode_set", self.on_conversation_mode_set)
        self.event_bus.subscribe("speech_started", self.on_speech_started)
        self.event_bus.subscribe("speech_finished", self.on_speech_finished)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.subscribe("speech_audio_level", self.on_speech_audio_level)
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe("agent_thinking_started", self.on_agent_thinking_started)
        self.event_bus.subscribe("agent_thinking_finished", self.on_agent_thinking_finished)
        self.event_bus.subscribe("task_activated", self.on_task_activated)
        self.event_bus.subscribe("task_completed", self.on_task_finished)
        self.event_bus.subscribe("task_failed", self.on_task_finished)
        self.event_bus.subscribe("task_paused", self.on_task_finished)
        self.event_bus.subscribe("task_cancelled", self.on_task_finished)
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

        if message_type == "hud.chat_delete":
            session_id = str(message.get("session_id") or "").strip()
            if self.chat_history.delete_session(session_id):
                memory = getattr(self.kernel, "memory", None)
                if memory is not None:
                    memory.delete_session(session_id)
                print(f"[Chat] Deleted session {session_id}", flush=True)
                self._publish_chat_context()
            return

        if message_type == "hud.chat_new":
            session_id = self.chat_history.new_session()
            print(f"[Chat] New session {session_id}", flush=True)
            sessions = self.chat_history.sessions()
            self.transport.publish_chat_history([], session_id=session_id, sessions=sessions)
            self.transport.publish_chat_sessions(sessions)
            return

        if message_type == "hud.conversation_mode":
            enabled = bool(message.get("enabled"))
            print(f"[HUD] Conversation mode requested: {'ON' if enabled else 'OFF'}", flush=True)
            self.event_bus.emit("conversation_mode_set", enabled=enabled)
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

    def on_incremental_user_message(self, text, *args, **kwargs):
        # Early command segments are intentionally separate from user_message
        # so AI/TaskRuntime can suppress mid-sentence acknowledgments without
        # changing the normal chat event contract. Keep the transcript visible
        # in the HUD regardless.
        if isinstance(text, str) and text.strip():
            value = text.strip()
            self.chat_history.append("user", value)
            self.transport.publish_chat(role="user", text=value)
            self._publish_chat_index()

    def on_incremental_voice_listening(self, enabled):
        if bool(enabled):
            self.set_state(
                mode="listening",
                intensity="high",
                status="LISTENING",
                progress=None,
                activity="active_utterance",
            )
            return

        task_manager = getattr(self.kernel, "task_manager", None)
        current = task_manager.current() if task_manager is not None else None
        if getattr(getattr(current, "status", None), "value", "") == "active":
            self.set_state(
                mode="executing",
                intensity="high",
                status="EXECUTING",
                progress=None,
                activity=getattr(current, "current_step", None) or "working",
            )

    def on_voice_action_committed(self, text="", *args, **kwargs):
        value = str(text or "").strip()
        if value:
            self.set_state(
                mode="executing",
                intensity="high",
                status="COMMITTING",
                progress=None,
                activity=value,
            )

    def on_user_message(self, text):
        if isinstance(text, str) and text.strip():
            value = text.strip()
            self.chat_history.append("user", value)
            self.transport.publish_chat(role="user", text=value)
            self._publish_chat_index()

        if self.state.mode in {"speaking", "approval", "executing"}:
            return

        try:
            intent = self.kernel.intent_router.analyze(text)
        except Exception:
            intent = None

        if getattr(intent, "intent", None) is not None and getattr(intent.intent, "value", "") == "command":
            self.set_state(
                mode="speaking",
                intensity="medium",
                status="ACKNOWLEDGING",
                progress=None,
                activity="acknowledgment",
            )
            return

        self.set_state(
            mode="thinking",
            intensity="high",
            status="THINKING",
            progress=None,
            activity="reasoning",
        )

    def on_command_acknowledged(self, text=None, *args, **kwargs):
        self.set_state(
            mode="speaking",
            intensity="medium",
            status="ACKNOWLEDGED",
            progress=None,
            activity="acknowledgment",
        )

    def on_task_progress(self, task_id=None, phase="working", text="", *args, **kwargs):
        phase_value = str(phase or "working").strip().lower()
        activity = str(text or "working").strip() or "working"

        if phase_value in {"planning", "replanning", "recovery"}:
            self.set_state(
                mode="thinking",
                intensity="high",
                status=phase_value.upper(),
                progress=None,
                activity=activity,
            )
        elif phase_value in {"executing", "planned"}:
            self.set_state(
                mode="executing",
                intensity="high",
                status="EXECUTING" if phase_value == "executing" else "READY",
                progress=None,
                activity=activity,
            )
        else:
            self.set_state(
                mode="thinking",
                intensity="medium",
                status=phase_value.upper(),
                progress=None,
                activity=activity,
            )

    def on_speech_started(self, *args, **kwargs):
        self.set_state(mode="speaking", intensity="high", status="SPEAKING", progress=None, activity="speech")
        self.transport.publish_audio_level(0.0)

    def on_speech_finished(self, *args, **kwargs):
        task = getattr(self.kernel, "task_manager", None)
        current = task.current() if task is not None else None
        status = getattr(current, "status", None)
        if getattr(status, "value", "") == "active":
            self.set_state(
                mode="executing",
                intensity="high",
                status="EXECUTING",
                progress=None,
                activity=getattr(current, "current_step", None) or "working",
            )
        elif getattr(status, "value", "") == "paused":
            self.set_state(
                mode="thinking",
                intensity="medium",
                status="PAUSED",
                progress=None,
                activity="waiting",
            )
        else:
            self.set_state(
                mode="listening",
                intensity="medium",
                status="LISTENING",
                progress=None,
                activity="command",
            )
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

    def on_agent_thinking_started(self, *args, **kwargs):
        self.set_state(
            mode="thinking",
            intensity="high",
            status="THINKING",
            progress=None,
            activity="agent_reasoning",
        )

    def on_agent_thinking_finished(self, *args, **kwargs):
        # Keep the current visual mode; the next tool_request/task event will
        # move the HUD to its authoritative execution or terminal state.
        return None

    def on_task_activated(self, *args, **kwargs):
        return None

    def on_task_finished(self, *args, **kwargs):
        self.set_state(
            mode="idle",
            intensity="low",
            status="IDLE",
            progress=None,
            activity=None,
        )

    def on_tool_confirmation_required(self, request=None, reason=None):
        self.set_state(mode="approval", intensity="high", status="APPROVAL", progress=None, activity=str(reason) if reason else "approval")

    def on_tool_confirmation_response(self, request_id=None, approved=False):
        if approved:
            self.set_state(mode="executing", intensity="high", status="EXECUTING", progress=None, activity="tool")
        else:
            self.set_state(mode="listening", intensity="medium", status="LISTENING", progress=None, activity="approval_rejected")

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
            ("command_acknowledged", self.on_command_acknowledged),
            ("task_progress", self.on_task_progress),
            ("user_message", self.on_user_message),
            ("conversation_mode_set", self.on_conversation_mode_set),
            ("speech_started", self.on_speech_started),
            ("speech_finished", self.on_speech_finished),
            ("speech_interrupt", self.on_speech_interrupt),
            ("speech_audio_level", self.on_speech_audio_level),
            ("tool_request", self.on_tool_request),
            ("agent_thinking_started", self.on_agent_thinking_started),
            ("agent_thinking_finished", self.on_agent_thinking_finished),
            ("task_activated", self.on_task_activated),
            ("task_completed", self.on_task_finished),
            ("task_failed", self.on_task_finished),
            ("task_paused", self.on_task_finished),
            ("task_cancelled", self.on_task_finished),
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
