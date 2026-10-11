import json
import time
from threading import RLock

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
        self._model_state_lock = RLock()
        self.transport = HUDTransport()
        self.chat_history = ChatHistoryStore()
        self._conversation_active = False
        self._runtime_ready = False

    def initialize(self):
        self.event_bus.subscribe("model_route_changed", self.on_model_route_changed)
        self.event_bus.subscribe("background_task_changed", self.on_background_task_changed)
        self.event_bus.subscribe("workspace_updated", self.on_workspace_updated)
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
            self._publish_workspace_context()
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

    def _publish_workspace_context(self, workspace=None):
        if not isinstance(workspace, dict):
            manager = getattr(self.kernel, "workspace_manager", None)
            workspace = manager.snapshot() if manager is not None else {}
        try:
            self.transport.publish_workspace_context(workspace)
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"[HUD] Workspace context publish failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def on_workspace_updated(self, workspace=None, **kwargs):
        self._publish_workspace_context(workspace)

    def on_model_route_changed(self, profile="", status="", model="", **kwargs):
        if profile not in {"general", "coding"} or status not in {"loading", "ready", "error", "generating", "stopped"}:
            return
        # Model loading must not overwrite approval or task-execution state.
        with self._model_state_lock:
            self.state.model_services = dict(self.state.model_services)
            self.state.model_services[profile] = {"status": status, "model": str(model or "").replace("\\", "/").rsplit("/", 1)[-1][:160]}
            self.state.model_profile = profile
            self.state.model_status = status
            self.state.model_name = str(model or "").replace("\\", "/").rsplit("/", 1)[-1][:160]
            self._publish_state()

    def on_background_task_changed(self, task=None, **kwargs):
        if not isinstance(task, dict):
            return
        # No file content, model reasoning or absolute workspace paths in status.
        self.state.background_task = {key: str(task.get(key) or "")[:200]
            for key in ("job_id", "task_id", "phase")}
        self._publish_state()

    def _background_owns(self, task_id=None):
        background = getattr(getattr(self, "kernel", None), "background_tasks", None)
        return background is not None and (background.owns(task_id) if task_id else background.in_worker())

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
        self._sync_action_display()
        try:
            self.transport.publish_state(self.state)
        except OSError as exc:
            print(f"[HUD] State publish failed: {type(exc).__name__}: {exc}", flush=True)

    def _sync_action_display(self):
        """Whitelist observable steps and exact approval requests, never plan content."""
        kernel = getattr(self, "kernel", None)
        runtime = getattr(kernel, "model_runtime", None)
        self.state.model_mode = getattr(runtime, "mode", None)
        background = getattr(kernel, "background_tasks", None)
        manager = getattr(kernel, "task_manager", None)
        snapshot = getattr(background, "snapshot", None)
        state = snapshot() if callable(snapshot) else None
        task_id = state.get("task_id") if state else None
        task = manager.get(task_id) if manager is not None and task_id else None
        self.state.project_steps = []
        if task is not None and task.plan is not None:
            for step in task.plan.steps:
                # Labels are built from registered tools, not model rationale/code.
                tool = str(step.metadata.get("tool") or "")
                paths = HUDTransport._safe_recent_files([step.metadata.get("path")])
                self.state.project_steps.append({"id": step.id, "tool": tool[:100],
                    "status": step.status.value, "path": paths[0] if paths else "",
                    "verification": str(step.metadata.get("verification") or "")[:80]})
        self.state.approvals = []
        approval_manager = getattr(kernel, "approval_manager", None)
        if approval_manager is not None:
            for pending in approval_manager.list_pending():
                request = pending.request
                preview = "Review the requested tool before approving."
                if request.tool == "system.run_command":
                    args = request.arguments
                    preview = json.dumps({"target": args.get("target"),
                        "arguments": args.get("arguments", []), "cwd": args.get("cwd", ".")}, ensure_ascii=False)
                ttl = approval_manager.ttl_seconds
                remaining = max(0, ttl - (time.monotonic() - pending.created_at))
                self.state.approvals.append({"request_id": request.request_id,
                    "tool": request.tool, "preview": preview,
                    "can_approve": request.tool == "system.run_command",
                    "expires_at": time.time() + remaining if ttl > 0 else None})

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
        if message_type == "hud.approval":
            request_id = message.get("request_id")
            approved = message.get("approved")
            manager = getattr(self.kernel, "approval_manager", None)
            pending = manager.get(request_id) if manager is not None and isinstance(request_id, str) else None
            if (isinstance(request_id, str) and isinstance(approved, bool)
                    and manager is not None and pending is not None and manager.contains(request_id)
                    and (not approved or pending.request.tool == "system.run_command")):
                self.event_bus.emit("tool_confirmation_response", request_id=request_id, approved=approved)
            self._publish_state()
            return

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
        if self._background_owns(task_id):
            self._publish_state()
            return  # The background row does not replace foreground voice state.
        phase_value = str(phase or "working").strip().lower()
        activity = str(text or "working").strip() or "working"
        sandbox_run = "isolated container" in activity.lower()

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
                status=(
                    "SANDBOX RUN"
                    if sandbox_run
                    else ("EXECUTING" if phase_value == "executing" else "READY")
                ),
                progress=None,
                activity=(
                    "Python/pytest · workspace snapshot is read-only · network off"
                    if sandbox_run
                    else activity
                ),
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
        if getattr(status, "value", "") == "active" and not self._background_owns(getattr(current, "id", None)):
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

    _SCREEN_TOOLS = frozenset({
        "vision.locate", "vision.inspect", "computer.click",
        "computer.type_text", "computer.keypress", "computer.hotkey",
        "computer.scroll", "computer.move_mouse",
    })

    def on_tool_request(self, request):
        if self._background_owns(getattr(request, "metadata", {}).get("task_id")):
            return
        tool = str(getattr(request, "tool", "") or "")
        sandbox_run = tool == "system.run_command"
        self.set_state(
            mode="executing",
            intensity="high",
            status="SANDBOX RUN" if sandbox_run else "EXECUTING",
            progress=None,
            activity=(
                "Python/pytest · workspace snapshot is read-only · network off"
                if sandbox_run
                else (tool or "tool")
            ),
        )
        if tool in self._SCREEN_TOOLS:
            self._step_aside()

    def _step_aside(self):
        """Minimize the HUD so screenshots and clicks reach the target app."""
        import os
        import time

        if os.getenv("ASTA_HUD_HIDE_DURING_GUI", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        if getattr(self, "_hud_hidden", False):
            return
        try:
            self.transport.publish_window("minimize")
        except (OSError, AttributeError):
            return
        self._hud_hidden = True
        # Let the minimize animation finish before the first screenshot.
        time.sleep(float(os.getenv("ASTA_HUD_HIDE_SETTLE", "0.45")))

    def _restore_window(self):
        if not getattr(self, "_hud_hidden", False):
            return
        self._hud_hidden = False
        try:
            self.transport.publish_window("restore")
        except (OSError, AttributeError):
            pass

    def on_agent_thinking_started(self, *args, **kwargs):
        if self._background_owns():
            return
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
        task = kwargs.get("task") or (args[0] if args else None)
        if isinstance(task, dict) and self._background_owns(task.get("id")):
            return
        self._restore_window()
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
            ("model_route_changed", self.on_model_route_changed),
            ("background_task_changed", self.on_background_task_changed),
            ("workspace_updated", self.on_workspace_updated),
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
