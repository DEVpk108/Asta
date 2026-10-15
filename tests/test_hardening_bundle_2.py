"""Deterministic session, memory, approval, and lifecycle regression tests.

No live model, credentials, desktop, audio hardware or MemPalace required.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import queue
import threading
import time

import pytest

from ai.ai_module import AIModule
from chat_history import ChatHistoryStore, shared_history
from core import Kernel
from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.event_bus import EventBus
from core.task_manager import TaskManager
from core.tools.approval import ApprovalManager, APPROVAL_EXPIRED_ERROR, PendingApproval
from core.tools.dispatcher import ToolDispatcher
from core.tools.module import ToolRuntimeModule
from core.tools.registry import ToolRegistry
from hud.hud_module import HUDModule
from memory.memory_module import MemoryModule
from memory.mempalace_adapter import MemPalaceAdapter


def spawn(fn):
    errors = []
    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def finish(job):
    thread, errors = job
    thread.join(3)
    assert not thread.is_alive(), 'worker did not finish'
    assert not errors, errors


def test_kernel_ai_hud_share_one_store_and_active_session(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTA_CHAT_HISTORY_DB', str(tmp_path / 'history.db'))
    kernel = Kernel()
    ai, hud = AIModule(kernel), HUDModule(kernel)
    assert ai.chat_history is hud.chat_history is kernel.chat_history
    store = kernel.chat_history
    store.initialize()
    connection = store._connection
    store.initialize()
    assert store._connection is connection
    old = store.new_session()
    store.append('assistant', 'older conversation')
    newer = store.new_session()
    store.append('assistant', 'newer conversation')
    assert store.switch_session(old)
    assert kernel.session_id == old != newer
    assert ai._active_chat_context(limit=5)[0]['text'] == 'older conversation'
    ai.shutdown()
    assert store._connection is connection  # kernel, not AI, owns lifetime
    kernel.shutdown()
    assert store._connection is None


def test_shared_store_construction_is_thread_safe():
    kernel = SimpleNamespace()
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = list(pool.map(lambda _: shared_history(kernel), range(32)))
    assert all(store is stores[0] for store in stores)


def test_store_serializes_initialization_append_and_close(tmp_path):
    store = ChatHistoryStore(tmp_path / 'history.db', history_limit=100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: store.initialize(), range(16)))
        list(pool.map(lambda n: store.append('user', str(n)), range(40)))
    assert len(store.recent()) == 40
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: store.close(), range(8)))
    assert store._connection is None


@pytest.mark.parametrize('change', ['hud.chat_new', 'hud.chat_select', 'hud.chat_delete'])
def test_hud_session_changes_emit_authoritative_id(tmp_path, change):
    store = ChatHistoryStore(tmp_path / 'history.db')
    store.initialize()
    target = store.new_session()
    store.append('user', 'target')
    if change == 'hud.chat_select':
        store.new_session()
    kernel = SimpleNamespace(event_bus=EventBus(), chat_history=store)
    hud = HUDModule(kernel)
    # No network transport or Electron needed for session routing.
    hud.transport = SimpleNamespace(publish_chat_history=lambda *a, **k: None,
        publish_chat_sessions=lambda *a, **k: None)
    hud._publish_chat_context = lambda: None
    events = []
    kernel.event_bus.subscribe('chat_session_changed', lambda session_id: events.append(session_id))
    hud.on_transport_message({'type': change, 'session_id': target})
    assert events == [store.session_id]
    assert (store.session_id == target) is (change == 'hud.chat_select')
    store.close()


def test_session_change_cancels_capture_and_resets_model():
    kernel = SimpleNamespace(event_bus=EventBus(), tool_registry=ToolRegistry())
    ai = AIModule(kernel)
    calls = []
    ai.engine = SimpleNamespace(cancel=lambda: calls.append('cancel'),
        reset_conversation=lambda: calls.append('reset'))
    ai._cancel_post_response_capture = lambda: calls.append('capture')
    ai.on_chat_session_changed('new')
    assert calls == ['capture', 'cancel', 'reset']


@pytest.mark.parametrize('switch', [False, True])
def test_completed_turn_is_full_once_and_old_session_results_are_dropped(switch):
    kernel = SimpleNamespace(event_bus=EventBus(), session_id='original', tool_registry=ToolRegistry())
    ai = AIModule(kernel)
    ai._prepare_runtime_context = lambda *a: ''
    events = []
    for event in ('assistant_sentence', 'assistant_response', 'conversation_turn_completed'):
        kernel.event_bus.subscribe(event, lambda event=event, **kw: events.append((event, kw)))
    def generate(text, on_sentence, context):
        if switch:
            kernel.session_id = 'new'
        on_sentence('First sentence.')
        on_sentence('Second sentence.')
        return 'First sentence. Second sentence.'
    ai.engine = SimpleNamespace(generate_response=generate)
    result = ai._generate_response('question')
    if switch:
        assert events == [] and result is None
    else:
        turns = [kw for name, kw in events if name == 'conversation_turn_completed']
        assert turns == [{'text': 'First sentence. Second sentence.', 'session_id': 'original'}]
        assert [name for name, _ in events].count('assistant_sentence') == 2


class Backend:
    available = True
    palace_path = 'fixture'
    error = None
    def __init__(self):
        self.records, self.deleted = [], []
        self.closed = 0
    def initialize(self):
        pass
    def remember(self, **item):
        assert not self.closed
        self.records.append(item)
    def context(self, *a, **kw):
        assert not self.closed
        return 'memory'
    def delete_session(self, session_id):
        self.deleted.append(session_id)
        self.records[:] = [r for r in self.records if r['session_id'] != session_id]
        return True
    def close(self):
        self.closed += 1


def memory_fixture():
    kernel = SimpleNamespace(event_bus=EventBus(), session_id='old')
    memory = MemoryModule(kernel)
    memory.backend = Backend()
    return kernel, memory


def test_memory_records_completed_turn_not_status_or_fragments():
    kernel, memory = memory_fixture()
    memory.initialize()
    try:
        kernel.event_bus.emit('assistant_sentence', text='Opening Chrome.')
        kernel.event_bus.emit('assistant_response', text='Task paused.')
        kernel.event_bus.emit('conversation_turn_completed', text='A complete response.', session_id='old')
        memory._write_queue.join()
        assert memory.backend.records == [dict(role='assistant', text='A complete response.', session_id='old')]
        kernel.memory_context = 'previous session'
        kernel.event_bus.emit('chat_session_changed', session_id='new')
        assert kernel.memory_context == ''
    finally:
        memory.shutdown()
    assert memory.backend.closed == 1


def test_queued_memory_captures_session_before_switch():
    kernel, memory = memory_fixture()
    memory.initialize()
    try:
        with memory._backend_lock:
            memory._queue_write('user', 'old message')
            kernel.session_id = 'new'
            memory._queue_write('user', 'new message')
        memory._write_queue.join()
        assert [(r['text'], r['session_id']) for r in memory.backend.records] == [
            ('old message', 'old'), ('new message', 'new')]
    finally:
        memory.shutdown()


def test_deleted_session_cannot_be_resurrected_by_queued_write():
    _, memory = memory_fixture()
    memory.initialize()
    try:
        with memory._backend_lock:
            memory._queue_write('user', 'delete me', session_id='old')
            memory.delete_session('old')
            memory._queue_write('user', 'retain me', session_id='new')
        memory._write_queue.join()
        assert [r['text'] for r in memory.backend.records] == ['retain me']
    finally:
        memory.shutdown()


def test_memory_shutdown_drains_before_single_close():
    _, memory = memory_fixture()
    memory.initialize()
    for i in range(30):
        memory._queue_write('user', str(i))
    memory.shutdown()
    memory.shutdown()
    memory._queue_write('user', 'too late')
    assert len(memory.backend.records) == 30
    assert memory.backend.closed == 1
    assert memory.recall('after close') == ''


def test_memory_backend_calls_never_overlap():
    _, memory = memory_fixture()
    entered, release, searched = threading.Event(), threading.Event(), threading.Event()
    original = memory.backend.remember
    def remember(**item):
        entered.set()
        assert release.wait(2)
        original(**item)
    memory.backend.remember = remember
    writer = spawn(lambda: memory._remember('user', 'text', 'old'))
    assert entered.wait(1)
    reader = spawn(lambda: (memory.recall('query'), searched.set()))
    try:
        assert not searched.wait(.05)
    finally:
        release.set()
        finish(writer)
        finish(reader)
    memory.shutdown()


@pytest.mark.parametrize('method,args,kwargs', [
    ('initialize', (), {}), ('remember', (), {'role': 'user', 'text': 'hello'}),
    ('delete_session', ('old',), {}), ('search', ('query',), {}),
    ('status', (), {}), ('close', (), {}),
])
def test_mempalace_adapter_public_operations_take_lock(monkeypatch, method, args, kwargs):
    import sys
    if method == 'initialize':
        monkeypatch.setitem(sys.modules, 'mempalace', None)
    adapter = MemPalaceAdapter()
    class Probe:
        entries = 0
        def __enter__(self):
            self.entries += 1
        def __exit__(self, *args):
            pass
    probe = adapter._lock = Probe()
    getattr(adapter, method)(*args, **kwargs)
    assert probe.entries >= 1


def approval_fixture():
    kernel = SimpleNamespace(event_bus=EventBus(), tool_registry=ToolRegistry(), approval_manager=ApprovalManager(ttl_seconds=60))
    dispatched, results = [], []
    kernel.tool_dispatcher = SimpleNamespace(dispatch=lambda r, **kw: (
        dispatched.append(r) or ToolResult(success=True, tool=r.tool)))
    runtime = ToolRuntimeModule(kernel)
    kernel.event_bus.subscribe('tool_result', lambda result: results.append(result))
    request = ToolRequest(tool='system.run_command', arguments={}, request_id='approval', metadata={'task_id': 'task'})
    kernel.approval_manager.request_approval(request, 'fixture')
    return kernel, runtime, request, dispatched, results


@pytest.mark.parametrize('approved', [True, False])
def test_duplicate_concurrent_approval_is_consumed_once(approved):
    kernel, runtime, request, dispatched, results = approval_fixture()
    barrier = threading.Barrier(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: (barrier.wait(), runtime._confirmation_response(request.request_id, approved)), range(8)))
    assert len(results) == 1
    assert len(dispatched) == int(approved)
    assert kernel.approval_manager.get(request.request_id) is None
    runtime.shutdown()


@pytest.mark.parametrize('approved', [True, False])
def test_late_approval_is_expired_failure_with_task_metadata(approved):
    kernel, runtime, request, dispatched, results = approval_fixture()
    kernel.approval_manager._pending[request.request_id] = PendingApproval(request, 'old', time.monotonic()-100)
    runtime._confirmation_response(request.request_id, approved)
    assert not dispatched
    assert len(results) == 1 and results[0].error == APPROVAL_EXPIRED_ERROR
    assert results[0].metadata['task_id'] == 'task'
    runtime.shutdown()


def test_approval_and_expiry_race_emits_one_terminal_result():
    kernel, runtime, request, dispatched, results = approval_fixture()
    kernel.approval_manager._pending[request.request_id] = PendingApproval(request, 'old', time.monotonic()-100)
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda approved: (barrier.wait(), runtime._confirmation_response(
            request.request_id, approved, expired=not approved)), [True, False]))
    assert len(results) == 1 and not dispatched
    runtime.shutdown()


def test_approval_timers_cancel_on_reply_shutdown_and_ignore_stale_identity():
    kernel, runtime, request, _, results = approval_fixture()
    runtime._schedule_expiry(request.request_id)
    timer = runtime._timers[request.request_id]
    runtime._expire_approval(request.request_id, timer=object())
    assert runtime._timers[request.request_id] is timer
    runtime._confirmation_response(request.request_id, False)
    assert timer.finished.is_set() and runtime._timers == {}
    runtime._schedule_expiry(request.request_id)  # immediate reply before schedule
    assert runtime._timers == {}
    kernel.approval_manager.request_approval(request, 'again')
    runtime._schedule_expiry(request.request_id)
    pending_timer = runtime._timers[request.request_id]
    runtime.shutdown()
    assert pending_timer.finished.is_set() and runtime._timers == {}
    runtime._expire_approval(request.request_id, timer=pending_timer)
    assert len(results) == 1


def test_expired_only_take_does_not_consume_live_approval():
    kernel, runtime, request, _, _ = approval_fixture()
    assert kernel.approval_manager.take(request.request_id, expired_only=True) is None
    assert kernel.approval_manager.contains(request.request_id)
    runtime.shutdown()


def test_desktop_tools_share_lock_but_other_work_can_continue():
    registry = ToolRegistry()
    entered, release, second, unrelated = [threading.Event() for _ in range(4)]
    def execute(request):
        if request.request_id == 'first':
            entered.set()
            assert release.wait(2)
        elif request.tool.startswith('computer.'):
            second.set()
        else:
            unrelated.set()
        return ToolResult(success=True, tool=request.tool)
    for name in ('computer.click', 'computer.type_text', 'filesystem.read_file'):
        registry.register(SimpleNamespace(definition=ToolDefinition(name, 'fixture', {}), execute=execute))
    dispatcher = ToolDispatcher(registry)
    first = spawn(lambda: dispatcher.dispatch(ToolRequest('computer.click', {}, 'first')))
    assert entered.wait(1)
    other = spawn(lambda: dispatcher.dispatch(ToolRequest('computer.type_text', {}, 'second')))
    read = spawn(lambda: dispatcher.dispatch(ToolRequest('filesystem.read_file', {}, 'read')))
    try:
        assert unrelated.wait(1)
        assert not second.wait(.05)
    finally:
        release.set()
        for job in (first, other, read):
            finish(job)
    assert second.is_set()


def test_desktop_serialization_does_not_bypass_authority():
    registry = ToolRegistry()
    called = []
    registry.register(SimpleNamespace(definition=ToolDefinition('computer.click', 'fixture', {},
        risk_level='critical', requires_confirmation=True), execute=lambda r: called.append(r)))
    result = ToolDispatcher(registry).dispatch(ToolRequest('computer.click', {}, 'denied'))
    assert not result.success and not called
    assert result.metadata['requires_confirmation']


def test_catalog_cache_is_bounded_and_returns_copies(tmp_path):
    from core.media.catalog import SongResolver
    resolver = SongResolver(fetch=lambda term, **kw: [dict(title=term, artist='fixture')], history_path=tmp_path/'h.json')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(resolver._search, map(str, range(300))))
    assert len(resolver._cache) == 256
    cached_key = next(iter(resolver._cache))
    resolver._search(cached_key)[0]['title'] = 'mutated'
    assert resolver._search(cached_key)[0]['title'] == cached_key


def test_terminal_task_retention_preserves_unfinished_tasks():
    manager = TaskManager(terminal_history_limit=3)
    paused = manager.create('retain paused')
    active = manager.create('retain active')
    for i in range(20):
        task = manager.create(str(i), activate=False)
        manager.complete(task_id=task.id)
    manager.create('prune boundary', activate=False)
    assert manager.get(paused.id) is paused and manager.get(active.id) is active
    assert len([t for t in manager.list() if t.status.value == 'completed']) == 3


@pytest.mark.parametrize('operation', ['generate_response', 'generate_for_task', 'complete_json'])
def test_model_runtime_admission_never_holds_client_lock(monkeypatch, operation):
    from tests.test_concurrent_models import dual
    from tests.test_model_router import FakeEngine
    from ai.model_router import RoutedLLMProvider
    _, runtime, _, _ = dual(monkeypatch)
    provider = RoutedLLMProvider(runtime, engine_factory=FakeEngine)
    original = runtime.execute
    probes = []
    def execute(*args, **kw):
        acquired = []
        def probe():
            ok = provider.client_lock.acquire(timeout=.2)
            acquired.append(ok)
            if ok:
                provider.client_lock.release()
        job = spawn(probe)
        finish(job)
        probes.extend(acquired)
        assert acquired == [True], 'client lock held during runtime admission'
        return original(*args, **kw)
    runtime.execute = execute
    try:
        if operation == 'generate_response':
            provider.generate_response('hello')
        elif operation == 'generate_for_task':
            provider.generate_for_task('prompt', goal='write Python')
        else:
            provider.complete_json('system', 'user')
        assert probes == [True]
    finally:
        runtime.shutdown()


def test_llama_cancel_does_not_wait_for_request_lock():
    from ai.llama_cpp_engine import LlamaCppEngine
    engine = LlamaCppEngine()
    with engine._request_lock:
        job = spawn(engine.cancel)
        finish(job)
        assert engine._cancel_event.is_set()
    engine.session.close()


@pytest.mark.parametrize('method,args', [
    ('reset_conversation', ()), ('set_system_prompt', ('fixture',)),
    ('generate_response', ('',)), ('complete_json', ('system', 'user')),
])
def test_llama_public_state_and_requests_are_serialized(monkeypatch, method, args):
    from ai.llama_cpp_engine import LlamaCppEngine
    engine = LlamaCppEngine()
    # Request method must block before even touching network/state.
    done = threading.Event()
    monkeypatch.setattr(engine.server_manager, 'ensure_running', lambda: False)
    monkeypatch.setattr(engine, '_discover_model', lambda: None)
    monkeypatch.setattr(engine.session, 'post', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('no network in fixture')))
    monkeypatch.setattr('ai.llama_cpp_engine.gpu_share.release_idle', lambda *a: None)
    # Call with lock held; errors after release are irrelevant to entry locking.
    def call():
        try:
            getattr(engine, method)(*args)
        except Exception:
            pass
        finally:
            done.set()
    with engine._request_lock:
        job = spawn(call)
        assert not done.wait(.05)
    finish(job)
    engine.session.close()


def test_sandbox_setup_helpers_are_safe_before_run():
    from core.autonomy.sandbox_setup import SandboxSetupOperator
    setup = SandboxSetupOperator()
    assert setup.executable is None
    assert not setup._engine_ready()
    assert not setup._image_present()


def speech_class_without_optional_audio():
    """Compile the actual class unchanged, excluding only optional audio import."""
    from core.module import Module
    path = Path(__file__).resolve().parents[1]/'speech'/'speech_module.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpeechModule')
    ns = dict(logger=__import__('logging').getLogger('speech.speech_module'), Module=Module, KokoroEngine=lambda: SimpleNamespace(), queue=queue,
        threading=threading, time=time, re=__import__('re'), unicodedata=__import__('unicodedata'))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['SpeechModule']


def test_speech_shutdown_defers_close_until_worker_exits():
    cls = speech_class_without_optional_audio()
    speech = cls(SimpleNamespace(event_bus=EventBus()))
    calls = []
    speech.engine = SimpleNamespace(abort=lambda: calls.append('abort'), close=lambda: calls.append('close'))
    # An uninterruptible device call exceeds the bounded join; do not close it.
    class Worker:
        alive = True
        def is_alive(self):
            return self.alive
        def join(self, timeout):
            calls.append(('join', timeout))
    worker = speech._speech_thread = Worker()
    speech.shutdown()
    assert calls == ['abort', ('join', 2.0)]
    speech._close_engine()  # worker finally owns close once it actually exits
    speech._close_engine()
    assert calls.count('close') == 1


def test_speech_worker_finally_closes_engine_once():
    cls = speech_class_without_optional_audio()
    speech = cls(SimpleNamespace(event_bus=EventBus()))
    calls = []
    speech.engine = SimpleNamespace(close=lambda: calls.append('close'))
    speech._running = True
    speech._queue.put(None)
    speech._speech_loop()
    speech._close_engine()
    assert calls == ['close']


def test_spotify_playback_uses_shared_401_refresh_path(monkeypatch):
    from core.media.providers import SpotifyProvider
    provider = SpotifyProvider()
    provider.client_id = 'fixture-only'
    tokens, requests = [], []
    monkeypatch.setattr(provider, '_get_access_token', lambda force_refresh=False: (
        tokens.append(force_refresh) or ('fresh' if force_refresh else 'expired')))
    monkeypatch.setattr(provider, '_invalidate_access_token', lambda: None)
    track = {'uri': 'spotify:track:fixture'}
    monkeypatch.setattr(provider, '_search_track', lambda *a: track)
    monkeypatch.setattr(provider, '_select_device', lambda *a: None)
    def request(method, url, **kwargs):
        requests.append((method, url, kwargs))
        return SimpleNamespace(status_code=401 if len(requests) == 1 else 204, text='')
    monkeypatch.setattr('core.media.providers.requests.request', request)
    assert provider._play_query('fixture') is track
    assert len(requests) == 2 and True in tokens
    assert requests[-1][2]['headers']['Authorization'] == 'Bearer fresh'
    assert all(r[0] == 'PUT' and r[1].endswith('/me/player/play') for r in requests)


@pytest.mark.parametrize('failure_at', ['goto', 'app_creation'])
def test_spotify_browser_context_closes_on_unexpected_failure(monkeypatch, tmp_path, failure_at):
    import sys
    from core.autonomy.spotify_setup import _PlaywrightSpotifyBrowser
    calls = []
    def fail():
        raise RuntimeError('fixture failure')
    page = SimpleNamespace(set_default_timeout=lambda *a: None,
        goto=lambda *a, **kw: fail() if failure_at == 'goto' else None,
        bring_to_front=lambda: None)
    context = SimpleNamespace(pages=[page], close=lambda: calls.append('close'))
    playwright = SimpleNamespace(chromium=SimpleNamespace(launch_persistent_context=lambda *a, **kw: context))
    class Manager:
        def __enter__(self):
            return playwright
        def __exit__(self, *a):
            pass
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', SimpleNamespace(
        TimeoutError=TimeoutError, sync_playwright=Manager))
    monkeypatch.setenv('ASTA_SPOTIFY_BROWSER_PROFILE', str(tmp_path/'profile'))
    browser = _PlaywrightSpotifyBrowser(dashboard_url='https://fixture.invalid', redirect_uri='fixture')
    monkeypatch.setattr(browser, '_wait_for_dashboard_render', lambda p: None)
    monkeypatch.setattr(browser, '_public_login_control_visible', lambda p: False)
    monkeypatch.setattr(browser, '_login_visible', lambda p: False)
    monkeypatch.setattr(browser, '_terms_gate_visible', lambda p: False)
    monkeypatch.setattr(browser, '_open_or_create_app', lambda p: fail())
    with pytest.raises(RuntimeError, match='fixture failure'):
        browser.run()
    assert calls == ['close']


@pytest.mark.parametrize('launch_fails', [False, True])
def test_project_program_log_handle_closes_on_every_exit(monkeypatch, tmp_path, launch_fails):
    from core.tools.project_program import OpenProjectProgramTool
    from core.workspace_manager import WorkspaceManager
    from tests.test_desktop_program import FakeProcess, GAME
    import tempfile
    workspace = WorkspaceManager()
    workspace.update_project(name='fixture', path=str(tmp_path))
    (tmp_path/'game.py').write_text(GAME)
    files = []
    real = tempfile.TemporaryFile
    def temporary(*a, **kw):
        handle = real(*a, **kw)
        files.append(handle)
        return handle
    monkeypatch.setattr('core.tools.project_program.tempfile.TemporaryFile', temporary)
    def popen(argv, **kw):
        if launch_fails:
            raise RuntimeError('fixture launch failure')
        return FakeProcess(0, '', kw['stderr'])
    tool = OpenProjectProgramTool(workspace, popen=popen, settle_seconds=0)
    result = tool.execute(ToolRequest('project.open_program', {'path': 'game.py'}, 'fixture'))
    assert len(files) == 1 and files[0].closed
    assert result.success is not launch_fails
    assert not list(tmp_path.glob('asta-program-*.log'))


def test_memory_shutdown_does_not_close_under_blocked_writer():
    _, memory = memory_fixture()
    memory.initialize()
    entered, release = threading.Event(), threading.Event()
    original = memory.backend.remember
    def blocked(**item):
        entered.set()
        assert release.wait(3)
        original(**item)
    memory.backend.remember = blocked
    memory._queue_write('user', 'pending')
    assert entered.wait(1)
    real_thread = memory._writer_thread
    # Model a timed-out join without imposing five seconds on the test suite.
    class Worker:
        def is_alive(self):
            return real_thread.is_alive()
        def join(self, timeout):
            pass
    memory._writer_thread = Worker()
    try:
        memory.shutdown()
        assert memory.backend.closed == 0
        assert memory.delete_session('old') is False
    finally:
        release.set()
        real_thread.join(3)
    assert not real_thread.is_alive()
    assert memory.backend.closed == 1
    assert len(memory.backend.records) == 1


def test_switching_away_and_back_invalidates_old_generation():
    kernel = SimpleNamespace(event_bus=EventBus(), session_id='original', tool_registry=ToolRegistry())
    ai = AIModule(kernel)
    ai._prepare_runtime_context = lambda *a: ''
    events = []
    kernel.event_bus.subscribe('assistant_response', lambda text: events.append(text))
    def generate(text, on_sentence, context):
        kernel.session_id = 'other'
        ai.on_chat_session_changed('other')
        kernel.session_id = 'original'
        ai.on_chat_session_changed('original')
        on_sentence('stale')
        return 'stale'
    ai.engine = SimpleNamespace(generate_response=generate)
    assert ai._generate_response('question') is None
    assert events == []


def test_kernel_signals_shutdown_completion_even_if_store_close_fails(monkeypatch):
    kernel = Kernel()
    monkeypatch.setattr(kernel.chat_history, 'close', lambda: (_ for _ in ()).throw(RuntimeError('close fixture')))
    with pytest.raises(RuntimeError, match='close fixture'):
        kernel.shutdown()
    assert kernel._shutdown_complete.is_set()
    assert kernel._shutdown_owner is None
