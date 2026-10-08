from types import SimpleNamespace

from core.event_bus import EventBus
from core.workspace_manager import WorkspaceManager
from hud.hud_module import HUDModule
from hud.transport import HUDTransport


class DummyRequest:
    tool = "run_command"


class RecordingHUDTransport(HUDTransport):
    def __init__(self):
        super().__init__(host="127.0.0.1", port=0, token="")
        self.messages = []

    def start(self):
        return None

    def _broadcast(self, message):
        self.messages.append(message)


def build_hud():
    kernel = SimpleNamespace(event_bus=EventBus())
    hud = HUDModule(kernel)
    hud.initialize()
    return kernel, hud


def test_hud_shows_only_safe_project_context_and_tracks_updates():
    event_bus = EventBus()
    manager = WorkspaceManager(event_bus=event_bus)
    manager.update_project(
        name="<ASTA & Project>",
        path=r"C:\Users\Private\Projects\ASTA",
        repository=(
            "https://bot:TOPSECRET@github.com/DEVpk108/Asta.git"
            "?access_token=also-secret#fragment"
        ),
        branch="feat/hud-context",
    )
    manager.set_recent_files(
        [
            "README.md",
            ".env",
            "../outside.py",
            r"C:\Users\Private\secret.py",
            "src/token.json",
            "docs/overview.md",
        ]
    )
    hud = HUDModule(SimpleNamespace(event_bus=event_bus, workspace_manager=manager))
    transport = RecordingHUDTransport()
    hud.transport = transport

    hud.initialize()
    messages = [item for item in transport.messages if item["type"] == "hud.workspace"]
    assert messages
    initial = messages[-1]["workspace"]
    assert initial == {
        "project_name": "<ASTA & Project>",
        "repository": "github.com/DEVpk108/Asta",
        "branch": "feat/hud-context",
        "recent_files": ["README.md", "docs/overview.md"],
    }
    assert "project_path" not in initial
    assert "TOPSECRET" not in str(initial)
    assert "also-secret" not in str(initial)

    manager.add_recent_file("core/agent/brain.py")
    updated = [
        item for item in transport.messages if item["type"] == "hud.workspace"
    ][-1]["workspace"]
    assert updated["recent_files"][0] == "core/agent/brain.py"

    hud.shutdown()
    count_after_shutdown = len(transport.messages)
    manager.add_recent_file("after/shutdown.py")
    assert len(transport.messages) == count_after_shutdown


def test_authenticated_hud_client_receives_cached_workspace_context():
    transport = RecordingHUDTransport()
    transport.publish_workspace_context(
        {
            "project_name": "ASTA",
            "project_path": r"C:\Users\Private\ASTA",
            "branch": "main",
            "recent_files": ["README.md"],
        }
    )
    sent = []
    transport._send_to_client = lambda client, message: sent.append(message)
    client = object()

    transport._register_authenticated_client(client)

    assert any(
        message["type"] == "hud.workspace"
        and message["workspace"]["project_name"] == "ASTA"
        for message in sent
    )
    assert all("project_path" not in message for message in sent)
    with transport._clients_lock:
        transport._clients.discard(client)
        transport._authenticated.discard(client)


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


def test_workspace_command_is_labeled_as_sandbox_execution():
    kernel, hud = build_hud()

    kernel.event_bus.emit(
        "tool_request",
        request=SimpleNamespace(tool="system.run_command"),
    )
    assert hud.get_state().status == "SANDBOX RUN"
    assert "read-only" in hud.get_state().activity
    assert "network off" in hud.get_state().activity

    kernel.event_bus.emit(
        "task_progress",
        task_id="task-1",
        phase="executing",
        text=(
            "Running Python in an isolated container. "
            "Only a sanitized, read-only workspace snapshot is mounted; "
            "network access is disabled."
        ),
    )
    assert hud.get_state().status == "SANDBOX RUN"
    assert hud.get_state().activity == (
        "Python/pytest · workspace snapshot is read-only · network off"
    )

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


def test_hud_text_input_uses_normal_user_message_event():
    kernel, hud = build_hud()

    hud.on_transport_message({
        "type": "hud.input",
        "version": 1,
        "input": {"text": "hello from HUD"},
    })

    assert hud.get_state().mode == "thinking"
    assert hud.get_state().activity == "reasoning"

    hud.shutdown()
