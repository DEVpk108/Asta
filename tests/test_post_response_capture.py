"""Deterministic timer races plus native AI/HUD protocol regression coverage."""
import ast
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from ai.ai_module import AIModule
from ai.final_runtime_patch import apply_final_runtime_patch
from ai.post_response_capture import PostResponseCapture
from core import EventBus
from core.contracts import IntentResult, IntentType, ToolRequest
from hud.hud_module import HUDModule
from hud.transport import HUDTransport


class Timer:
    def __init__(self, delay, callback):
        self.delay, self.callback, self.cancelled = delay, callback, False
    def start(self):
        pass
    def cancel(self):
        self.cancelled = True
    def fire(self):
        # Deliberately allow firing cancelled timers: models already-running callbacks.
        self.callback()


class Timers(list):
    def __call__(self, delay, callback):
        timer = Timer(delay, callback)
        self.append(timer)
        return timer


def make_capture():
    bus, timers = EventBus(), Timers()
    capture = PostResponseCapture(bus, timers)
    requests, protocol = [], []
    bus.subscribe('tool_request', lambda request: requests.append(request))
    for phase in ('begin', 'ready', 'cancel'):
        bus.subscribe('capture_response_' + phase, lambda phase=phase, **data: protocol.append((phase, data)))
    request = ToolRequest(tool='vision.screenshot', arguments={}, request_id='test-capture')
    return capture, bus, timers, requests, protocol, request


def prepared():
    capture, bus, timers, requests, protocol, request = make_capture()
    token = capture.begin(request)
    capture.complete(token, 'Complete answer.')
    return capture, bus, timers, requests, protocol, request, token


def test_ack_waits_for_settle_and_dispatches_once():
    capture, bus, timers, requests, protocol, request, token = prepared()
    assert timers[0].delay == 1.5 and not requests
    bus.emit('hud_rendered', response_id=token)
    assert timers[0].cancelled and timers[1].delay == .2 and not requests
    bus.emit('hud_rendered', response_id=token)
    timers[0].fire(); timers[1].fire(); timers[1].fire()
    assert requests == [request] and len(timers) == 2
    assert request.metadata['capture_render_status'] == 'hud_acknowledged'
    assert request.metadata['capture_response_id'] == token
    capture.close()


def test_timeout_is_labelled_not_misrepresented_as_rendered():
    capture, bus, timers, requests, protocol, request, token = prepared()
    timers[0].fire()
    bus.emit('hud_rendered', response_id=token)
    timers[1].fire()
    assert requests == [request]
    assert request.metadata['capture_render_status'] == 'timeout_fallback'
    capture.close()


@pytest.mark.parametrize('ack', [None, '', 'old-token', 123, False, []])
def test_unrelated_ack_cannot_arm_a_capture(ack):
    capture, bus, timers, requests, _, _, _ = prepared()
    bus.emit('hud_rendered', response_id=ack)
    assert len(timers) == 1 and not requests
    capture.close()


def test_ack_during_generation_is_not_a_completion():
    capture, bus, timers, requests, _, request = make_capture()
    token = capture.begin(request)
    bus.emit('hud_rendered', response_id=token)
    assert not timers and not requests
    capture.complete(token, 'Answer')
    assert len(timers) == 1
    capture.close()


@pytest.mark.parametrize('phase', ['generating', 'waiting', 'settling'])
@pytest.mark.parametrize('operation', ['cancel', 'close'])
def test_cancel_and_shutdown_neutralize_even_running_timer_callbacks(phase, operation):
    capture, bus, timers, requests, _, request = make_capture()
    token = capture.begin(request)
    if phase != 'generating': capture.complete(token, 'Answer')
    if phase == 'settling': bus.emit('hud_rendered', response_id=token)
    getattr(capture, operation)()
    for timer in list(timers): timer.fire()
    capture.complete(token, 'Late answer')
    bus.emit('hud_rendered', response_id=token)
    assert not requests
    capture.close()
    assert bus.subscriber_count('hud_rendered') == 0
    assert capture.begin(request) is None


@pytest.mark.parametrize('response', ['', None])
def test_empty_response_does_not_capture(response):
    capture, _, timers, requests, protocol, request = make_capture()
    token = capture.begin(request)
    capture.complete(token, response)
    assert not timers and not requests and protocol[-1][0] == 'cancel'
    capture.close()


def test_new_response_supersedes_old_callbacks_and_stale_completion():
    capture, bus, timers, requests, _, old, old_token = prepared()
    new = ToolRequest(tool='vision.screenshot', arguments={}, request_id='test-capture')
    token = capture.begin(new)
    capture.complete(old_token, 'stale response')
    capture.complete(token, 'new response')
    bus.emit('hud_rendered', response_id=old_token)
    timers[0].fire()
    bus.emit('hud_rendered', response_id=token)
    timers[-1].fire()
    assert requests == [new]
    capture.close()


def test_ack_timeout_race_is_exactly_once():
    for _ in range(30):
        capture, bus, timers, requests, _, request, token = prepared()
        barrier = threading.Barrier(3)
        def run(action):
            barrier.wait(); action()
        threads = [threading.Thread(target=run, args=(action,)) for action in (
            lambda: bus.emit('hud_rendered', response_id=token), timers[0].fire)]
        for thread in threads: thread.start()
        barrier.wait()
        for thread in threads: thread.join(2); assert not thread.is_alive()
        assert len(timers) == 2
        timers[1].fire()
        assert requests == [request]
        capture.close()


def test_close_waits_for_inflight_dispatch_without_holding_event_lock():
    capture, bus, timers, requests, _, _, token = prepared()
    bus.emit('hud_rendered', response_id=token)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    bus.subscribe('tool_request', lambda request: (entered.set(), release.wait(2)))
    worker = threading.Thread(target=timers[1].fire)
    worker.start(); assert entered.wait(2)
    closer = threading.Thread(target=lambda: (capture.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(.03)
    finally:
        release.set(); worker.join(2); closer.join(2)
    assert closed.is_set() and not worker.is_alive() and not closer.is_alive()


def test_reentrant_close_from_cancel_event_prevents_tool_emission():
    capture, bus, timers, requests, _, _, token = prepared()
    bus.emit('hud_rendered', response_id=token)
    bus.subscribe('capture_response_cancel', lambda response_id: capture.close())
    timers[1].fire()
    assert not requests


def fake_ai(response='Answer.'):
    bus, timers = EventBus(), Timers()
    kernel = SimpleNamespace(event_bus=bus, background_tasks=None,
        intent_router=SimpleNamespace(analyze=lambda text: IntentResult(intent=IntentType.CONVERSATION,
            confidence=.8, normalized_text=text, entities={})),
        context_builder=SimpleNamespace(build_prompt=lambda text, intent: 'context'))
    ai = AIModule.__new__(AIModule)
    ai.kernel, ai.event_bus = kernel, bus
    ai._post_response_capture = PostResponseCapture(bus, timers)
    ai.tool_request_builder = SimpleNamespace(build=lambda intent: ToolRequest(tool='vision.screenshot', arguments={}, request_id='test-capture'))
    def generate(text, on_sentence, context):
        on_sentence('Answer.'); return response
    ai.engine = SimpleNamespace(generate_response=generate, cancel=lambda: None)
    ai._handle_approval_response = lambda text: False
    return ai, bus, timers


def test_native_path_renders_completed_response_before_request():
    ai, bus, timers = fake_ai()
    events, requests = [], []
    bus.subscribe('assistant_sentence', lambda text: events.append(('sentence', text)))
    bus.subscribe('assistant_response', lambda text: events.append(('response', text)))
    bus.subscribe('capture_response_ready', lambda response_id, text: events.append(('ready', text)))
    bus.subscribe('tool_request', lambda request: requests.append(request))
    ai.on_user_message('Tell me a joke then take a screenshot')
    assert events == [('sentence', 'Answer.'), ('response', 'Answer.'), ('ready', 'Answer.')]
    assert not requests
    token = ai._post_response_capture._pending['id']
    bus.emit('hud_rendered', response_id=token); timers[-1].fire()
    assert len(requests) == 1 and requests[0].metadata['post_response_action'] is True
    ai._post_response_capture.close()


@pytest.mark.parametrize('mode', ['empty', 'model_error', 'markup'])
def test_generation_failure_or_tool_markup_never_schedules_capture(mode):
    ai, bus, timers = fake_ai('' if mode == 'empty' else 'Answer.')
    if mode == 'model_error':
        ai.engine.generate_response = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('model'))
        with pytest.raises(RuntimeError, match='model'):
            ai._handle_post_response_screenshot('Tell a joke then take a screenshot')
    else:
        if mode == 'markup':
            ai.engine.generate_response = lambda *args, **kwargs: "<|tool_call_start|>[system.run_command(command='unsafe')]<|tool_call_end|>"
        ai._handle_post_response_screenshot('Tell a joke then take a screenshot')
    assert not timers and ai._post_response_capture._pending is None
    ai._post_response_capture.close()


def test_consent_handler_runs_before_native_mixed_request():
    ai, _, timers = fake_ai()
    ai._handle_approval_response = lambda text: True
    ai.on_user_message('Tell a joke then take a screenshot')
    assert not timers and ai._post_response_capture._pending is None
    ai._post_response_capture.close()


def test_compound_tool_sequence_remains_router_owned():
    ai, _, timers = fake_ai()
    ai.kernel.intent_router.analyze = lambda text: IntentResult(intent=IntentType.COMMAND,
        confidence=1., normalized_text=text, entities={'commands': [{}, {}]})
    assert ai._handle_post_response_screenshot('Open Chrome then take a screenshot') is False
    assert not timers
    ai._post_response_capture.close()


def test_speech_interrupt_cancels_pending_capture():
    ai, _, timers = fake_ai()
    ai._handle_post_response_screenshot('Tell a joke then take a screenshot')
    ai.on_speech_interrupt()
    assert timers[0].cancelled and ai._post_response_capture._pending is None
    ai._post_response_capture.close()


def test_shutdown_closes_capture_before_engine_and_history():
    ai, bus, timers = fake_ai()
    ai._handle_post_response_screenshot('Tell a joke then take a screenshot')
    order = []
    ai.chat_history = SimpleNamespace(close=lambda: order.append(ai._post_response_capture._closed))
    ai.kernel.decision_engine = SimpleNamespace()
    ai.engine.shutdown = lambda: order.append(ai._post_response_capture._closed)
    ai.shutdown()
    assert order == [True, True] and timers[0].cancelled
    assert bus.subscriber_count('hud_rendered') == 0


def test_real_legacy_ai_hook_delegates_mixed_request_to_native(monkeypatch):
    from ai.runtime_patch import apply_ai_runtime_patch
    for name in ('on_user_message', 'on_tool_result', '_is_creator_identity_question', '_format_tool_success'):
        monkeypatch.setattr(AIModule, name, AIModule.__dict__[name])
    monkeypatch.setattr(AIModule, '_asta_runtime_patch_applied', False, raising=False)
    voice = ModuleType('voice.voice_module')
    voice.VoiceModule = type('VoiceModule', (), {'on_conversation_mode_set': lambda self, *a, **k: None,
                                                '_can_listen': lambda self: True})
    monkeypatch.setitem(sys.modules, 'voice.voice_module', voice)
    apply_ai_runtime_patch()
    ai, bus, timers = fake_ai()
    requests=[]; bus.subscribe('tool_request', lambda request: requests.append(request))
    ai.on_user_message('Tell me a joke then take a screenshot')
    assert len(timers) == 1 and not requests
    timers[0].fire(); timers[1].fire()
    assert len(requests) == 1
    ai._post_response_capture.close()


def test_retired_final_hook_is_idempotent_and_startup_does_not_call_it():
    before = AIModule.on_user_message
    apply_final_runtime_patch(); apply_final_runtime_patch()
    assert AIModule.on_user_message is before
    tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.Name) and node.id == 'apply_final_runtime_patch' for node in ast.walk(tree))


def test_hud_forwards_ephemeral_protocol_and_validates_ack_without_changing_chat():
    bus = EventBus()
    hud = HUDModule(SimpleNamespace(event_bus=bus))
    sent, ack = [], []
    hud.transport._broadcast = lambda message: sent.append(message)
    bus.subscribe('hud_rendered', lambda response_id: ack.append(response_id))
    token = 'a' * 32
    hud.on_capture_response_begin(token)
    hud.on_capture_response_ready(token, 'Full response')
    hud.on_capture_response_cancel(token)
    assert [item['phase'] for item in sent] == ['begin', 'ready', 'cancel']
    assert all(item['type'] == 'hud.capture_response' for item in sent)
    for invalid in (None, 32, 'wrong', 'A' * 32, 'a' * 33):
        hud.on_transport_message({'type': 'hud.rendered', 'response_id': invalid})
    assert not ack
    hud.on_transport_message({'type': 'hud.rendered', 'response_id': token})
    assert ack == [token]
    assert hud.transport._last_chat_history_message is None


def test_capture_protocol_is_not_cached_for_reconnect():
    transport = HUDTransport(token='test')
    sent=[];transport._broadcast=lambda message: sent.append(message)
    transport.publish_capture_response('ready', 'a'*32, 'response')
    replay=[];transport._send_to_client=lambda client, message: replay.append(message)
    transport._register_authenticated_client(object())
    assert not any(item.get('type') == 'hud.capture_response' for item in replay)


def test_renderer_and_electron_protocol_with_node():
    node = shutil.which('node')
    if node is None: pytest.skip('Node is needed for Electron protocol tests')
    result = subprocess.run([node, '--test', 'tests/js/test_capture_response.cjs'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_generation_through_hud_ack_and_real_tool_dispatcher():
    from core import Kernel
    from core.tools import ScreenshotTool, ToolRequestBuilder, ToolRuntimeModule
    kernel = Kernel()
    shots=[];kernel.register_tool(ScreenshotTool(capture=lambda: shots.append('fake.png') or {'path': 'fake.png'}))
    ai, _, timers = fake_ai()
    ai._post_response_capture.close()
    ai.kernel, ai.event_bus = kernel, kernel.event_bus
    kernel.context_builder=SimpleNamespace(build_prompt=lambda text, intent: 'context')
    ai.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)
    ai._post_response_capture = PostResponseCapture(kernel.event_bus, timers)
    hud=HUDModule(kernel)
    packets=[];hud.transport._broadcast=lambda message: packets.append(message)
    for phase in ('begin', 'ready', 'cancel'):
        kernel.event_bus.subscribe('capture_response_'+phase, getattr(hud, 'on_capture_response_'+phase))
    runtime=ToolRuntimeModule(kernel);runtime.initialize()
    results=[];kernel.event_bus.subscribe('tool_result', lambda result: results.append(result))
    try:
        assert ai._handle_post_response_screenshot('Tell me a joke then take a screenshot')
        ready=next(packet for packet in packets if packet.get('phase') == 'ready')
        assert not shots
        hud.on_transport_message({'type': 'hud.rendered', 'response_id': ready['response_id']})
        timers[-1].fire();timers[0].fire();timers[-1].fire()
        assert shots == ['fake.png']
        assert len(results) == 1 and results[0].success is True
    finally:
        ai._post_response_capture.close();runtime.shutdown();kernel.shutdown()


def test_request_builder_failure_leaves_no_pending_timer():
    ai, _, timers = fake_ai()
    replies=[];ai._emit_assistant_text=replies.append
    ai.tool_request_builder.build=lambda intent: (_ for _ in ()).throw(ValueError('not registered'))
    assert ai._handle_post_response_screenshot('Tell a joke then take a screenshot') is True
    assert not timers and ai._post_response_capture._pending is None
    assert 'not registered' in replies[0]
    ai._post_response_capture.close()


def test_new_native_turn_cancels_old_capture_before_consent_handling():
    ai, _, timers = fake_ai()
    ai._handle_post_response_screenshot('Tell a joke then take a screenshot')
    observations=[]
    ai._handle_approval_response=lambda text: observations.append(ai._post_response_capture._pending) or True
    ai.on_user_message('cancel')
    assert observations == [None] and timers[0].cancelled
    ai._post_response_capture.close()


def test_hud_subscribes_and_unsubscribes_capture_protocol(tmp_path):
    from chat_history import ChatHistoryStore
    bus=EventBus();hud=HUDModule(SimpleNamespace(event_bus=bus))
    hud.chat_history=ChatHistoryStore(db_path=tmp_path/'capture-history.db')
    hud.transport.start=lambda: None
    hud.transport.stop=lambda: None
    packets=[];hud.transport._broadcast=lambda message: packets.append(message)
    hud.initialize()
    for phase in ('begin', 'ready', 'cancel'):
        assert bus.subscriber_count('capture_response_'+phase) == 1
    bus.emit('capture_response_ready', response_id='a'*32, text='Answer')
    assert any(item.get('phase') == 'ready' for item in packets)
    hud.shutdown()
    for phase in ('begin', 'ready', 'cancel'):
        assert bus.subscriber_count('capture_response_'+phase) == 0
