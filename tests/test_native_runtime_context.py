"""Context preparation belongs to native generation, not startup patch order."""
import ast
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from ai.ai_module import AIModule
from ai.context_runtime_patch import apply_context_runtime_patch
from ai.runtime_patch import apply_ai_runtime_patch
from core import EventBus


class Router:
    def __init__(self):
        self.calls = []
        self.intent = object()
        self.fail = False

    def analyze(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError('router fixture failed')
        return self.intent


class Builder:
    def __init__(self):
        self.calls = []
        self.fail = False

    def build_prompt(self, text, intent):
        self.calls.append((text, intent))
        if self.fail:
            raise RuntimeError('builder fixture failed')
        return f'RUNTIME: {text}'


class Engine:
    def __init__(self, response='First sentence. Second sentence.'):
        self.calls = []
        self.response = response

    def generate_response(self, text, *, on_sentence, context):
        self.calls.append((text, context))
        if self.response:
            on_sentence('First sentence.')
            on_sentence('Second sentence.')
        return self.response


def _make(response='First sentence. Second sentence.'):
    kernel = SimpleNamespace(event_bus=EventBus(), intent_router=Router(), context_builder=Builder())
    # No provider, SQLite store, optional audio backend or module startup needed.
    ai = AIModule.__new__(AIModule)
    ai.kernel, ai.event_bus, ai.engine = kernel, kernel.event_bus, Engine(response)
    return ai, kernel


@pytest.fixture(autouse=True)
def restore_class_state(monkeypatch):
    for name in ('_generate_response', 'on_user_message', 'on_tool_result',
                 '_is_creator_identity_question', '_format_tool_success'):
        monkeypatch.setattr(AIModule, name, AIModule.__dict__[name])
    for name in ('_asta_runtime_patch_applied', '_asta_context_runtime_patch_applied'):
        monkeypatch.setattr(AIModule, name, False, raising=False)


def test_native_generation_enriches_without_a_startup_hook():
    ai, kernel = _make()
    ai._generate_response('remember my project')
    assert ai.engine.calls == [('remember my project', 'RUNTIME: remember my project')]
    assert kernel.intent_router.calls == ['remember my project']
    assert kernel.context_builder.calls == [('remember my project', kernel.intent_router.intent)]


@pytest.mark.parametrize('context', [None, '', False, 0, [], {}])
def test_falsy_explicit_context_keeps_existing_rebuild_policy(context):
    ai, kernel = _make()
    ai._generate_response('request', runtime_context=context)
    assert ai.engine.calls == [('request', 'RUNTIME: request')]
    assert len(kernel.intent_router.calls) == len(kernel.context_builder.calls) == 1


@pytest.mark.parametrize('context', ['caller context', {'opaque': True}, ['opaque']])
def test_truthy_explicit_context_is_forwarded_unchanged_without_routing(context):
    ai, kernel = _make()
    ai._generate_response('request', runtime_context=context)
    assert ai.engine.calls[0][1] is context
    assert kernel.intent_router.calls == kernel.context_builder.calls == []


def test_existing_builder_is_reused_and_builds_once_per_turn():
    ai, kernel = _make()
    builder = kernel.context_builder
    ai._generate_response('one')
    ai._generate_response('two')
    assert kernel.context_builder is builder
    assert [item[0] for item in builder.calls] == ['one', 'two']
    assert ai.engine.calls == [('one', 'RUNTIME: one'), ('two', 'RUNTIME: two')]


def test_missing_builder_is_created_once_and_cached(monkeypatch):
    ai, kernel = _make()
    del kernel.context_builder
    builder, creations = Builder(), []
    module = importlib.import_module('ai.ai_module')
    monkeypatch.setattr(module, 'ContextBuilder',
                        lambda owner: creations.append(owner) or builder, raising=False)
    ai._generate_response('one')
    ai._generate_response('two')
    assert creations == [kernel]
    assert kernel.context_builder is builder and len(builder.calls) == 2


@pytest.mark.parametrize('component', ['intent_router', 'context_builder'])
def test_context_enrichment_failure_falls_back_to_contextless_generation(component, capsys):
    ai, kernel = _make()
    getattr(kernel, component).fail = True
    ai._generate_response('request')
    assert ai.engine.calls == [('request', None)]
    assert 'Failed to build runtime context' in capsys.readouterr().out


def test_streaming_precedes_completed_response_with_native_context():
    ai, kernel = _make()
    events = []
    kernel.event_bus.subscribe('assistant_sentence', lambda text: events.append(('sentence', text)))
    kernel.event_bus.subscribe('assistant_response', lambda text: events.append(('response', text)))
    ai._generate_response('request')
    assert events == [('sentence', 'First sentence.'), ('sentence', 'Second sentence.'),
                      ('response', 'First sentence. Second sentence.')]
    assert ai.engine.calls[0][1] == 'RUNTIME: request'


@pytest.mark.parametrize('response', ['', None])
def test_empty_generation_does_not_emit_a_completed_response(response):
    ai, kernel = _make(response)
    completed = []
    kernel.event_bus.subscribe('assistant_response', lambda text: completed.append(text))
    ai._generate_response('request')
    assert completed == [] and len(ai.engine.calls) == 1


def test_engine_errors_are_not_swallowed_as_context_failures():
    ai, kernel = _make()
    def broken_engine(*args, **kwargs):
        raise RuntimeError('model fixture failed')
    ai.engine.generate_response = broken_engine
    with pytest.raises(RuntimeError, match='model fixture failed'):
        ai._generate_response('request')


@pytest.mark.parametrize('repetitions', [1, 3])
def test_legacy_context_hook_does_not_replace_native_generation(repetitions):
    original = AIModule._generate_response
    for _ in range(repetitions):
        assert apply_context_runtime_patch() is None
        assert AIModule._generate_response is original


def test_main_no_longer_imports_or_calls_context_patch():
    tree = ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'ai.context_runtime_patch'
                   for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'apply_context_runtime_patch' for node in ast.walk(tree))


def test_active_ai_patch_no_longer_contains_duplicate_generation_or_assignment():
    tree = ast.parse((Path(__file__).resolve().parents[1]/'ai/runtime_patch.py').read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.FunctionDef) and node.name == 'patched_generate_response'
                   for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute) and target.attr == '_generate_response'
        for target in node.targets) for node in ast.walk(tree))


@pytest.mark.parametrize('context_hook_first', [True, False])
def test_real_ai_patch_preserves_native_generation_in_either_legacy_call_order(monkeypatch, context_hook_first):
    # Substitute only the optional audio dependency; execute the real AI hook.
    class Voice:
        def on_conversation_mode_set(self, enabled):
            self._conversation_active = enabled
        def _can_listen(self):
            return True
    package, voice_module = ModuleType('voice'), ModuleType('voice.voice_module')
    package.__path__ = []
    voice_module.VoiceModule = Voice
    monkeypatch.setitem(sys.modules, 'voice', package)
    monkeypatch.setitem(sys.modules, 'voice.voice_module', voice_module)
    original = AIModule._generate_response
    if context_hook_first:
        apply_context_runtime_patch()
    apply_ai_runtime_patch()
    if not context_hook_first:
        apply_context_runtime_patch()
    apply_ai_runtime_patch()  # existing idempotent hook contract
    assert AIModule._generate_response is original
    ai, kernel = _make()
    ai._generate_response('request')
    assert ai.engine.calls == [('request', 'RUNTIME: request')]
    assert len(kernel.intent_router.calls) == len(kernel.context_builder.calls) == 1
