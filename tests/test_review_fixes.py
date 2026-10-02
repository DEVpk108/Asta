"""Regression tests for the code-review hardening fixes."""

import json
import socket
import time

import pytest

from core.contracts import ToolRequest
from core.intent_router import IntentRouter
from core.tools.approval import ApprovalManager
from hud.transport import HUDTransport


# ---------------------------------------------------------------- HUD socket

def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _send(port, payload: bytes):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
        client.sendall(payload)
        time.sleep(0.4)


@pytest.fixture
def transport():
    received = []
    hud = HUDTransport(host="127.0.0.1", port=_free_port(), token="secret-token")
    hud.set_command_handler(received.append)
    hud.start()
    time.sleep(0.1)
    yield hud, received
    hud.stop()


def test_transport_rejects_cross_protocol_http_request(transport):
    hud, received = transport
    body = json.dumps({"type": "hud.input", "input": {"text": "press windows r"}})
    request = (
        "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\n\r\n"
        f"\n{body}\n"
    ).encode()
    _send(hud.port, request)
    assert received == []


def test_transport_rejects_commands_without_valid_hello(transport):
    hud, received = transport
    bad_hello = json.dumps({"type": "hud.hello", "token": "wrong"}) + "\n"
    command = json.dumps({"type": "hud.input", "input": {"text": "hi"}}) + "\n"
    _send(hud.port, (bad_hello + command).encode())
    _send(hud.port, command.encode())
    assert received == []


def test_transport_accepts_authenticated_client(transport):
    hud, received = transport
    hello = json.dumps({"type": "hud.hello", "token": "secret-token"}) + "\n"
    command = json.dumps({"type": "hud.input", "input": {"text": "hi"}}) + "\n"
    _send(hud.port, (hello + command).encode())
    assert received == [{"type": "hud.input", "input": {"text": "hi"}}]


# ---------------------------------------------------------------- approvals

def _request():
    import uuid

    return ToolRequest(
        request_id=uuid.uuid4().hex,
        tool="system.run_command",
        arguments={"target": "x"},
    )


def test_expired_approval_cannot_be_approved():
    manager = ApprovalManager(ttl_seconds=0.05)
    request = _request()
    manager.request_approval(request, "critical")
    assert manager.list_pending()
    time.sleep(0.1)
    assert manager.list_pending() == ()
    assert [item.request for item in manager.list_expired()] == [request]
    assert manager.approve(request.request_id) is None


def test_keyboard_tools_require_confirmation():
    from core.tools import (
        ComputerController,
        ComputerHotkeyTool,
        ComputerKeypressTool,
        ComputerTypeTextTool,
    )

    controller = ComputerController()
    for tool in (
        ComputerTypeTextTool(controller),
        ComputerKeypressTool(controller),
        ComputerHotkeyTool(controller),
    ):
        assert tool.definition.risk_level == "high"


# ---------------------------------------------------------------- AI module

class _FakeEngine:
    system_prompt = ""

    def __init__(self):
        self.calls = []

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def generate_response(self, text, on_sentence=None, context=None):
        self.calls.append(text)
        if on_sentence:
            on_sentence("LLM answer.")
        return "LLM answer."


def _ai(tmp_path):
    from ai.ai_module import AIModule
    from core import Kernel
    from core.tools import RunCommandTool, ToolRuntimeModule

    kernel = Kernel(authority_path=tmp_path / "authority.json")
    kernel.register_tool(RunCommandTool())
    tools = ToolRuntimeModule(kernel)
    tools.initialize()
    ai = AIModule(kernel)
    ai.engine = _FakeEngine()
    ai.chat_history.db_path = tmp_path / "chat.db"
    ai.initialize()
    spoken = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text: spoken.append(text))
    return kernel, ai, spoken


def test_can_you_question_reaches_the_llm_not_the_tool_list(tmp_path):
    kernel, ai, spoken = _ai(tmp_path)
    ai.on_user_message("can you tell me a joke")
    assert ai.engine.calls == ["can you tell me a joke"]
    assert not any("registered capabilities" in text for text in spoken)


def test_remember_request_gets_a_spoken_reply(tmp_path):
    kernel, ai, spoken = _ai(tmp_path)
    ai.on_user_message("remember that my exam is on monday")
    assert spoken, "memory requests must not be silent"


def test_bare_ok_does_not_approve_and_new_request_cancels_pending(tmp_path):
    kernel, ai, spoken = _ai(tmp_path)
    responses = []
    kernel.event_bus.subscribe(
        "tool_confirmation_response",
        lambda request_id, approved: responses.append(approved),
    )
    request = _request()
    kernel.approval_manager.request_approval(request, "critical")

    ai.on_user_message("ok")
    assert responses == [False]
    assert kernel.approval_manager.list_pending() == ()


def test_creator_identity_patch_works_on_current_python(tmp_path):
    from ai.runtime_patch import apply_ai_runtime_patch

    apply_ai_runtime_patch()
    kernel, ai, spoken = _ai(tmp_path)
    ai.on_user_message("what is the capital of france")
    assert ai.engine.calls


def test_speech_interrupt_cancels_generation(tmp_path):
    kernel, ai, spoken = _ai(tmp_path)
    cancelled = []
    ai.engine.cancel = lambda: cancelled.append(True)
    kernel.event_bus.emit("speech_interrupt")
    assert cancelled == [True]


# ---------------------------------------------------------------- LLM engine

def test_llm_history_is_bounded(monkeypatch):
    from ai.llama_cpp_engine import LlamaCppEngine

    monkeypatch.setenv("ASTA_LLM_HISTORY_TURNS", "3")
    engine = LlamaCppEngine(base_url="http://127.0.0.1:9/v1")
    for index in range(10):
        engine._messages.append({"role": "user", "content": f"q{index}"})
        engine._messages.append({"role": "assistant", "content": f"a{index}"})
        engine._trim_history()
    assert engine._messages[0]["role"] == "system"
    assert len(engine._messages) == 1 + 3 * 2
    assert engine._messages[1] == {"role": "user", "content": "q7"}


# ---------------------------------------------------------------- routing

@pytest.mark.parametrize(
    "text",
    [
        "tell me how to open excel files",
        "run me through the plan",
        "start a timer for 5 minutes",
        "next question",
    ],
)
def test_conversational_sentences_are_not_commands(text):
    assert IntentRouter().analyze(text).intent.value != "command"


def test_stop_the_music_pauses_media():
    result = IntentRouter().analyze("stop the music")
    assert result.entities == {"action": "media", "operation": "pause"}


def test_open_up_chrome_for_me_targets_chrome():
    result = IntentRouter().analyze("open up chrome for me")
    assert result.entities == {"action": "open", "target": "chrome"}


# ---------------------------------------------------------------- incremental

class _Apps:
    names = ["Google Chrome", "Microsoft Excel", "Weather", "News", "Calculator"]

    def resolve(self, target):
        from core.applications.manager import ApplicationManager, ApplicationRecord

        manager = ApplicationManager()
        manager._applications = tuple(
            ApplicationRecord(name, name, "test") for name in self.names
        )
        manager._last_refresh = time.monotonic()
        return manager.resolve(target)


def _commits(text):
    from voice.incremental_command_engine import IncrementalCommandDetector

    detector = IncrementalCommandDetector(application_manager=_Apps(), stable_updates=1)
    words = text.split()
    commits = []
    for end in range(2, len(words) + 1):
        commit = detector.observe(" ".join(words[:end]))
        if commit is not None:
            commits.append(commit.text)
    return commits


@pytest.mark.parametrize(
    "text",
    [
        "tell me how to open excel files in python",
        "i want to start a new project",
        "open the door",
    ],
)
def test_incremental_detector_ignores_non_commands(text):
    assert _commits(text) == []


def test_incremental_detector_does_not_match_substrings():
    assert _commits("can you open the news about cricket") == ["open News"]


def test_incremental_detector_still_commits_real_commands():
    assert _commits("hey asta open up chrome for me and search") == ["open Google Chrome"]
