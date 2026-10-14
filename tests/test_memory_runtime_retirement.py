"""Retire the memory pass-through wrapper without changing the memory pipeline."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

from ai.ai_module import AIModule
from ai.context_runtime_patch import apply_context_runtime_patch
from core import Kernel
from memory import MemoryModule
from memory import runtime_patch as compatibility


@pytest.fixture(autouse=True)
def restore_patch_state(monkeypatch):
    # Protect global method state while characterizing the old/new hooks.
    monkeypatch.setattr(AIModule, '_generate_response', AIModule._generate_response)
    monkeypatch.setattr(AIModule, '_asta_context_runtime_patch_applied', False, raising=False)
    for attribute in ('_PATCHED', '_ORIGINAL_GENERATE_RESPONSE'):
        if hasattr(compatibility, attribute):
            monkeypatch.setattr(compatibility, attribute,
                False if attribute == '_PATCHED' else getattr(compatibility, attribute))


@pytest.mark.parametrize('repetitions', [1, 2, 5])
def test_compatibility_hook_never_replaces_generation(repetitions):
    original = AIModule._generate_response
    for _ in range(repetitions):
        assert compatibility.apply_memory_runtime_patch() is None
        assert AIModule._generate_response is original


def test_compatibility_hook_does_not_restore_or_capture_a_later_replacement(monkeypatch):
    compatibility.apply_memory_runtime_patch()
    replacement = lambda self, text, runtime_context=None: 'replacement'
    monkeypatch.setattr(AIModule, '_generate_response', replacement)
    compatibility.apply_memory_runtime_patch()
    assert AIModule._generate_response is replacement


@pytest.mark.parametrize('context', [None, '', 'explicit runtime context', {'opaque': 'context'}])
def test_existing_generation_receives_identical_arguments_and_return_value(monkeypatch, context):
    calls = []
    sentinel = object()

    def generate(self, text, runtime_context=None):
        calls.append((self, text, runtime_context))
        return sentinel

    monkeypatch.setattr(AIModule, '_generate_response', generate)
    compatibility.apply_memory_runtime_patch()
    instance = object()
    assert AIModule._generate_response(instance, 'request', runtime_context=context) is sentinel
    assert calls == [(instance, 'request', context)]
    assert calls[0][2] is context


def test_main_no_longer_imports_or_calls_retired_memory_hook():
    source = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'memory.runtime_patch'
                   for node in ast.walk(tree))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'apply_memory_runtime_patch' for node in ast.walk(tree))


def test_compatibility_module_can_load_without_importing_ai():
    # Load the real compatibility file, isolating its own dependency contract
    # from the memory package's unrelated eager exports and optional backends.
    path = Path(compatibility.__file__).resolve()
    script = """
import runpy, sys
sys.modules['ai'] = None
module = runpy.run_path(sys.argv[1])
assert module['apply_memory_runtime_patch']() is None
"""
    result = subprocess.run([sys.executable, '-c', script, str(path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


class FakeMemoryBackend:
    def __init__(self, available):
        self.available = available
        self.palace_path = 'inert-fixture'
        self.error = None
        self.calls = []
        self.records = []
        self.closed = False

    def initialize(self):
        return self.available

    def context(self, text, *, n_results):
        self.calls.append(('recall', text, n_results))
        return 'The user prefers local models.' if self.available else ''

    def remember(self, *, role, text, session_id):
        self.calls.append(('remember', role, text, session_id))
        if self.available:
            self.records.append((role, text, session_id))
        return self.available

    def close(self):
        self.closed = True


class StreamingEngine:
    def __init__(self):
        self.calls = []

    def generate_response(self, text, on_sentence=None, context=None):
        self.calls.append((text, context))
        on_sentence('I can help with that.')
        return 'I can help with that.'


@pytest.mark.parametrize('call_legacy_hook', [False, True])
@pytest.mark.parametrize('available,text,explicit_context,expect_memory', [
    (True, 'Do you remember my project?', None, True),
    (False, 'Do you remember my project?', None, False),
    (True, 'What is photosynthesis?', None, False),
    (True, 'Do you remember my project?', 'explicit caller context', False),
])
def test_memory_recall_context_streaming_and_writes_survive_retirement(
    monkeypatch, call_legacy_hook, available, text, explicit_context, expect_memory,
):
    kernel = Kernel()
    memory = MemoryModule(kernel)
    backend = FakeMemoryBackend(available)
    memory.backend = backend
    monkeypatch.setattr(memory, '_current_session_id', lambda: 'inert-session')
    ai = AIModule(kernel)
    engine = StreamingEngine()
    ai.engine = engine
    events = []
    kernel.event_bus.subscribe('assistant_sentence', lambda text: events.append(('sentence', text)))
    kernel.event_bus.subscribe('assistant_response', lambda text: events.append(('response', text)))
    apply_context_runtime_patch()  # retained owner of prompt assembly
    generation = AIModule._generate_response
    if call_legacy_hook:
        compatibility.apply_memory_runtime_patch()
    memory.initialize()
    # Same recall-before-generation order as main's module registration. Use
    # generation directly so this test does not change intent/approval routing.
    callback = lambda text: ai._generate_response(text, runtime_context=explicit_context)
    kernel.event_bus.subscribe('user_message', callback)
    try:
        kernel.event_bus.emit('user_message', text)
        memory._write_queue.join()
        assert len(engine.calls) == 1
        assert engine.calls[0][0] == text
        context = engine.calls[0][1]
        if explicit_context:
            assert context == explicit_context
        else:
            assert 'A.S.T.A. RUNTIME CONTEXT' in context
            assert ('LONG-TERM MEMORY:' in context) is expect_memory
            if expect_memory:
                assert context.count('The user prefers local models.') == 1
        assert backend.calls[0] == ('recall', text, 5)
        assert events == [('sentence', 'I can help with that.'),
                          ('response', 'I can help with that.')]
        expected = [('user', text, 'inert-session'),
                    ('assistant', 'I can help with that.', 'inert-session')]
        assert backend.records == (expected if available else [])
        # The retained context wrapper still owns generation; the compatibility
        # hook must not add another wrapper around it.
        if call_legacy_hook:
            assert AIModule._generate_response is generation
    finally:
        kernel.event_bus.unsubscribe('user_message', callback)
        memory.shutdown()
        ai.shutdown()
        kernel.shutdown()
    assert backend.closed and kernel.memory_context == ''
