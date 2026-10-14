"""Consent regression tests without microphones, models or real tool execution.

The voice adapter is compiled from its real source method because importing the
voice package eagerly loads optional audio engines. No audio module is mocked.
The AI approval handler and ApprovalManager are exercised directly.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.ai_module import AIModule
from core.approval_language import normalize_confirmation_transcript
from core.contracts import ToolRequest
from core.tools.approval import ApprovalManager


def _voice_normalizer():
    tree = ast.parse((Path(__file__).parents[1] / "voice" / "voice_module.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VoiceModule")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_normalize_confirmation_transcript")
    method.decorator_list = []
    namespace = {"normalize_confirmation_transcript": normalize_confirmation_transcript}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual-voice-adapter>", "exec"), namespace)
    return namespace[method.name]


@pytest.fixture
def approval():
    manager = ApprovalManager()
    request = ToolRequest(request_id="consent-regression", tool="system.run_command",
                          arguments={"target": "python", "arguments": ["example.py"]},
                          metadata={"task_id": "project-1"})
    manager.request_approval(request, "Critical command needs explicit consent")
    events = []
    ai = object.__new__(AIModule)
    ai.kernel = SimpleNamespace(approval_manager=manager, background_tasks=None)
    ai.event_bus = SimpleNamespace(emit=lambda event, **payload: events.append((event, payload)))
    ai._emit_assistant_text = lambda text: events.append(("said", {"text": text}))
    return ai, manager, events


def _decisions(events):
    return [payload for event, payload in events if event == "tool_confirmation_response"]


@pytest.mark.parametrize("text", [
    "okay", "ok", "sure", "okay stop", "okay wait a second", "okay don't do it",
    "yes cancel", "yes but don't do it", "yeah no", "yes no", "yes, not now",
    "yes don't", "yes don’t", "yes do not proceed", "yes wait", "yes never mind",
    "हाँ मत करो", "haan nahi", "confirm nothing", "I cannot approve", "do it later",
    "yes?", "I confirm?", "yes, don't proceed", "yes, don’t proceed",
])
def test_voice_refusals_and_ambiguous_replies_never_approve(text, approval):
    ai, manager, events = approval
    ai._handle_approval_response(_voice_normalizer()(text))
    assert _decisions(events)
    assert all(item["approved"] is False for item in _decisions(events))


@pytest.mark.parametrize("text", [
    "yes", "Yes!", "yeah go ahead", "yes please do it", "I confirm", "i just said yes",
    "okay go ahead", "sure go ahead", "go ahead", "हाँ", "haan", "no thanks",
    "cancel it", "don't do it", "stop", "approve coding step", "reject project step",
])
def test_voice_preserves_the_whole_confirmation_reply(text):
    normalized = _voice_normalizer()(text)
    # No first-word or prefix reduction is allowed: background approval phrases
    # and any subsequent negation must reach the handler intact.
    assert normalized == " ".join(text.lower().split()).rstrip(" .!;:,")


@pytest.mark.parametrize("text", [
    "yes", "Yes!", "yeah go ahead", "yes please do it", "I confirm", "i just said yes",
    "okay go ahead", "sure go ahead", "go ahead", "हाँ", "haan",
])
def test_explicit_voice_approvals_still_work(text, approval):
    ai, manager, events = approval
    assert ai._handle_approval_response(_voice_normalizer()(text)) is True
    assert _decisions(events) == [{"request_id": "consent-regression", "approved": True}]


@pytest.mark.parametrize("text", [
    "yes don’t do it", "yes don’t", "yes don't", "yes no", "yes, not now",
    "I cannot approve", "confirm nothing", "yes open chrome", "I said yes yesterday",
    "should I say yes", "yes?", "yes if the tests pass", "I might confirm",
])
def test_typed_uncertain_or_negated_replies_never_approve(text, approval):
    ai, manager, events = approval
    ai._handle_approval_response(text)
    assert _decisions(events)
    assert all(item["approved"] is False for item in _decisions(events))


@pytest.mark.parametrize("text, expected", [
    ("i just said yes", (True, False)), ("yes please do it", (True, False)),
    ("हाँ", (True, False)), ("no don't", (False, True)),
    ("okay", (False, False)), ("yes no", (False, False)),
])
def test_existing_short_reply_contract(text, expected):
    assert AIModule._classify_short_approval(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("approve coding step", True), ("approve project step", True),
    ("reject coding step", False), ("reject project step", False),
])
def test_background_approval_phrases_survive_voice_path(text, expected, approval):
    ai, manager, events = approval
    ai.kernel.background_tasks = SimpleNamespace(owns=lambda task_id: task_id == "project-1")
    assert ai._handle_approval_response(_voice_normalizer()(text)) is True
    assert _decisions(events) == [{"request_id": "consent-regression", "approved": expected}]


@pytest.mark.parametrize("text", ["yes", "okay", "sure", "yes please do it", "okay don't do it"])
def test_background_request_cannot_be_approved_by_generic_voice_reply(text, approval):
    ai, manager, events = approval
    ai.kernel.background_tasks = SimpleNamespace(owns=lambda task_id: task_id == "project-1")
    ai._handle_approval_response(_voice_normalizer()(text))
    assert not any(item["approved"] for item in _decisions(events))
    assert manager.contains("consent-regression")


@pytest.mark.parametrize("text", [
    "approve coding step?", "approve project step if the tests pass",
    "approve coding step, don't", "reject coding step, actually approve",
])
def test_background_approval_must_be_an_unambiguous_complete_command(text, approval):
    ai, manager, events = approval
    ai.kernel.background_tasks = SimpleNamespace(owns=lambda task_id: task_id == "project-1")
    ai._handle_approval_response(_voice_normalizer()(text))
    assert not any(item["approved"] for item in _decisions(events))


@pytest.mark.parametrize("text, expected_calls", [
    ("yes please do it", 1), ("okay don't do it", 0), ("okay stop", 0),
    ("sure", 0), ("yes don’t", 0), ("yes open chrome", 0),
])
def test_voice_reply_reaches_real_runtime_and_executes_at_most_once(text, expected_calls):
    from core import Kernel
    from core.contracts import ToolDefinition, ToolResult
    from core.tools.module import ToolRuntimeModule

    calls = []

    class InertCriticalTool:
        definition = ToolDefinition(
            name="test.consent", description="Inert consent test",
            input_schema={"type": "object", "properties": {}},
            risk_level="critical",
        )

        def execute(self, request):
            calls.append(request.request_id)
            return ToolResult(success=True, tool=request.tool)

    kernel = Kernel()
    kernel.register_tool(InertCriticalTool())
    runtime = ToolRuntimeModule(kernel)
    runtime.initialize()
    ai = object.__new__(AIModule)
    ai.kernel, ai.event_bus = kernel, kernel.event_bus
    ai._emit_assistant_text = lambda text: None
    request = ToolRequest(request_id="inert-runtime-test", tool="test.consent", arguments={})
    try:
        kernel.event_bus.emit("tool_request", request=request)
        assert calls == []
        assert kernel.approval_manager.contains(request.request_id)
        ai._handle_approval_response(_voice_normalizer()(text))
        assert len(calls) == expected_calls
        assert not kernel.approval_manager.contains(request.request_id)
        # A duplicate approval cannot run a consumed/rejected request.
        kernel.event_bus.emit("tool_confirmation_response",
                              request_id=request.request_id, approved=True)
        assert len(calls) == expected_calls
    finally:
        runtime.shutdown()


def test_expired_request_cannot_be_revived_by_explicit_voice_approval(approval, monkeypatch):
    ai, manager, events = approval
    pending = manager.list_pending()[0]
    monkeypatch.setattr("core.tools.approval.time.monotonic",
                        lambda: pending.created_at + manager.ttl_seconds + 1)
    ai._handle_approval_response(_voice_normalizer()("yes please do it"))
    assert _decisions(events) == [{"request_id": "consent-regression", "approved": False}]
    assert manager.approve("consent-regression") is None
