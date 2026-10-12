from __future__ import annotations

import threading
from types import SimpleNamespace

from ai.model_router import ModelRuntime, RoutedLLMProvider
from core.background_tasks import BackgroundTaskCoordinator
from core.contracts import IntentResult, IntentType
from core.event_bus import EventBus
from core.task_manager import TaskManager
from core.tools.approval import ApprovalManager
from tests.test_model_router import FakeEngine, FakeManager


def dual(monkeypatch):
    monkeypatch.setenv("ASTA_MODEL_MODE", "dual")
    log = []
    kernel = SimpleNamespace(event_bus=EventBus())
    managers = {p: FakeManager(p, log) for p in ("general", "coding")}
    runtime = ModelRuntime(kernel, managers=managers)
    kernel.model_runtime = runtime
    return kernel, runtime, managers, log


def test_chat_completes_while_coding_inference_is_blocked(monkeypatch):
    kernel, runtime, managers, log = dual(monkeypatch)
    chat = RoutedLLMProvider(runtime, engine_factory=FakeEngine)
    coder = RoutedLLMProvider(runtime, purpose="agent", engine_factory=FakeEngine)
    entered, release, answered = threading.Event(), threading.Event(), threading.Event()
    error = []
    engine = coder._engine("coding")
    original = engine.generate_response
    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    engine.generate_response = blocking
    def code():
        try:
            coder.generate_for_task("task prompt", goal="Debug Python", coding_scope=True)
        except Exception as exc:
            error.append(exc)
    thread = threading.Thread(target=code)
    thread.start()
    assert entered.wait(2)
    def converse():
        assert chat.generate_response("Explain what Python is") == "generated: Explain what Python is"
        assert chat.complete_json("system", "user") == '{"action":"none"}'
        answered.set()
    foreground = threading.Thread(target=converse)
    foreground.start()
    try:
        assert answered.wait(1), "chat was serialized behind coding"
        assert thread.is_alive()
        assert all(m.running for m in managers.values())
        assert not any(s.startswith("stop:") for s in log)
        assert coder._engine("coding")._messages[0]["content"] == coder.system_prompt
    finally:
        release.set()
        thread.join(3); foreground.join(3); runtime.shutdown()
    assert not error


def test_coding_waits_at_boundary_for_foreground_turn(monkeypatch):
    _, runtime, _, _ = dual(monkeypatch)
    chat = RoutedLLMProvider(runtime, engine_factory=FakeEngine)
    coder = RoutedLLMProvider(runtime, purpose="agent", engine_factory=FakeEngine)
    general_entered, release, coding_entered = threading.Event(), threading.Event(), threading.Event()
    def general(_engine):
        general_entered.set(); assert release.wait(3)
    thread = threading.Thread(target=lambda: runtime.execute("general", chat._engine("general"), general))
    thread.start(); assert general_entered.wait(2)
    worker = threading.Thread(target=lambda: runtime.execute("coding", coder._engine("coding"), lambda _: coding_entered.set()))
    worker.start()
    try:
        assert not coding_entered.wait(0.1)
        release.set()
        assert coding_entered.wait(2)
    finally:
        release.set(); thread.join(3); worker.join(3); runtime.shutdown()


def test_dual_missing_coder_preserves_conversation(monkeypatch):
    _, runtime, managers, _ = dual(monkeypatch)
    chat = RoutedLLMProvider(runtime, engine_factory=FakeEngine)
    coder = RoutedLLMProvider(runtime, purpose="agent", engine_factory=FakeEngine)
    assert chat.warmup()
    managers['coding'].valid = False
    import pytest
    with pytest.raises(Exception, match="unavailable"):
        coder.generate_for_task("prompt", goal="Debug Python")
    assert managers['general'].running
    assert chat.generate_response("Hello") == "generated: Hello"
    runtime.shutdown()


def background():
    bus = EventBus()
    kernel = SimpleNamespace(event_bus=bus, task_manager=TaskManager(event_bus=bus),
        approval_manager=ApprovalManager(), model_runtime=None)
    coordinator = BackgroundTaskCoordinator(kernel)
    kernel.background_tasks = coordinator
    return kernel, coordinator


def task_intent():
    return IntentResult(intent=IntentType.TASK, confidence=1.0,
        normalized_text="Debug Python", entities={})


def test_background_start_returns_and_reserves_single_task_slot():
    kernel, coordinator = background()
    entered, release = threading.Event(), threading.Event()
    def starter(goal, intent):
        entered.set(); assert release.wait(3)
        task = kernel.task_manager.create(goal)
        kernel.task_manager.complete(task_id=task.id)
        return task
    kernel.task_runtime = SimpleNamespace(start_goal=starter)
    try:
        assert coordinator.start("Debug Python", task_intent())
        assert entered.wait(2)
        assert coordinator.busy
        assert not coordinator.start("Other project", task_intent())
        assert coordinator.handle_control("project task status")
    finally:
        release.set(); coordinator.queue.join(); coordinator.shutdown()
    assert coordinator.snapshot()['phase'] == 'completed'


def test_cancel_during_planning_prevents_new_tools():
    kernel, coordinator = background()
    entered, release = threading.Event(), threading.Event()
    def starter(goal, intent):
        entered.set(); assert release.wait(3)
        return kernel.task_manager.create(goal)
    kernel.task_runtime = SimpleNamespace(start_goal=starter)
    try:
        coordinator.start("Debug Python", task_intent()); assert entered.wait(2)
        assert coordinator.cancel()
        release.set(); coordinator.queue.join()
        task = kernel.task_manager.get(coordinator.snapshot()['task_id'])
        assert task.status.value == 'cancelled'
        from core.contracts import ToolRequest
        assert not coordinator.allow_request(ToolRequest(tool='workspace.read_file', request_id='cancel-test', arguments={}, metadata={'task_id':task.id}))
    finally:
        release.set(); coordinator.shutdown()


def test_approval_continuation_runs_on_worker_not_chat_thread():
    kernel, coordinator = background()
    from core.contracts import ToolRequest, ToolResult
    from core.tools.module import ToolRuntimeModule
    kernel.tool_registry = SimpleNamespace()
    tool_module = ToolRuntimeModule(kernel)
    executed = threading.Event()
    threads = []
    def dispatch(request, confirmed=False):
        threads.append(threading.current_thread().name)
        assert confirmed
        executed.set()
        return ToolResult(success=True, tool=request.tool, output={})
    kernel.tool_dispatcher = SimpleNamespace(dispatch=dispatch)
    kernel.tool_registry = SimpleNamespace()
    kernel.task_runtime = SimpleNamespace(start_goal=lambda goal, intent: kernel.task_manager.create(goal))
    try:
        coordinator.start('Debug Python', task_intent()); coordinator.queue.join()
        task_id = coordinator.snapshot()['task_id']
        request = ToolRequest(tool='system.run_command', request_id='approval-test', arguments={}, metadata={'task_id':task_id})
        kernel.approval_manager.request_approval(request=request, reason='critical')
        tool_module.on_confirmation_response(request.request_id, True)
        assert executed.wait(2)
        coordinator.queue.join()
        assert threads == ['asta-project-worker']
    finally:
        coordinator.shutdown()


def test_background_approval_survives_chat_but_plain_yes_cannot_execute():
    kernel, coordinator = background()
    from ai.ai_module import AIModule
    from core.contracts import ToolRequest
    from core import Kernel
    # Test only approval interpretation, not model/network initialization.
    ai = AIModule.__new__(AIModule)
    ai.kernel = kernel; ai.event_bus = kernel.event_bus
    replies, approvals = [], []
    ai._emit_assistant_text = replies.append
    kernel.event_bus.subscribe('tool_confirmation_response', lambda **kw: approvals.append(kw))
    kernel.task_runtime = SimpleNamespace(start_goal=lambda goal, intent: kernel.task_manager.create(goal))
    try:
        coordinator.start('Debug Python', task_intent()); coordinator.queue.join()
        task_id = coordinator.snapshot()['task_id']
        request = ToolRequest(tool='system.run_command', request_id='preserve', arguments={}, metadata={'task_id':task_id})
        kernel.approval_manager.request_approval(request, 'critical')
        assert not ai._handle_approval_response('What is the capital of India?')
        assert kernel.approval_manager.get('preserve') is not None
        assert ai._handle_approval_response('yes')
        assert not approvals
        assert ai._handle_approval_response('approve coding step')
        assert approvals == [{'request_id':'preserve', 'approved':True}]
    finally:
        coordinator.shutdown()


def test_failed_background_task_rejects_stale_approval_and_tool():
    kernel, coordinator = background()
    from core.contracts import ToolRequest
    kernel.task_runtime = SimpleNamespace(start_goal=lambda goal, intent: kernel.task_manager.create(goal))
    try:
        coordinator.start('Debug Python', task_intent()); coordinator.queue.join()
        task_id = coordinator.snapshot()['task_id']
        request = ToolRequest(tool='system.run_command', request_id='stale', arguments={}, metadata={'task_id':task_id})
        kernel.approval_manager.request_approval(request, 'critical')
        kernel.task_manager.fail('failure', task_id)
        assert not kernel.approval_manager.list_pending()
        assert not coordinator.allow_request(request)
    finally:
        coordinator.shutdown()


def test_workspace_change_cancels_background_task():
    kernel, coordinator = background()
    from core.workspace_manager import WorkspaceManager
    kernel.workspace_manager = WorkspaceManager(event_bus=kernel.event_bus)
    kernel.workspace_manager.update_project(path='/first')
    kernel.task_runtime = SimpleNamespace(start_goal=lambda goal, intent: kernel.task_manager.create(goal))
    try:
        coordinator.start('Debug Python', task_intent()); coordinator.queue.join()
        kernel.workspace_manager.update_project(path='/second')
        assert coordinator.snapshot()['phase'] == 'cancelled'
    finally:
        coordinator.shutdown()


def test_conflicting_request_does_not_rebind_background_task():
    from core import Kernel
    from core.task_runtime import TaskRuntimeModule
    from core.tools.module import ToolRuntimeModule
    from core.contracts import ToolRequest
    kernel = Kernel()
    task_runtime = TaskRuntimeModule(kernel); task_runtime.initialize()
    tools = ToolRuntimeModule(kernel); tools.initialize()
    coordinator = BackgroundTaskCoordinator(kernel); kernel.background_tasks = coordinator
    # Use a stub only for planning; actual bus callbacks validate ownership.
    kernel.task_runtime = SimpleNamespace(start_goal=lambda goal, intent: kernel.task_manager.create(goal))
    try:
        coordinator.start('Debug Python', task_intent()); coordinator.queue.join()
        task_id = coordinator.snapshot()['task_id']
        request = ToolRequest(tool='workspace.read_file', request_id='conflict', arguments={'path':'a.py'})
        kernel.event_bus.emit('tool_request', request=request)
        assert 'task_id' not in request.metadata
        assert kernel.task_manager.get(task_id).status.value == 'active'
    finally:
        coordinator.shutdown(); tools.shutdown(); task_runtime.shutdown(); kernel.shutdown()


def test_real_project_bus_keeps_chat_available_then_continues_after_approval(tmp_path):
    from tests.test_workspace_project_task import _build_runtime, _happy_plan, _shutdown, GOAL
    workspace = tmp_path/'project'; workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path/'authority.json', _happy_plan(), authorize_run=False)
    coordinator = BackgroundTaskCoordinator(kernel); kernel.background_tasks = coordinator
    planning, release = threading.Event(), threading.Event()
    original_plan = kernel.agent_brain.plan
    def plan(*args, **kwargs):
        planning.set(); assert release.wait(3)
        return original_plan(*args, **kwargs)
    kernel.agent_brain.plan = plan
    messages = []
    kernel.event_bus.subscribe('assistant_response', lambda text=None, **kw: messages.append(text))
    class Conversation:
        def generate_response(self, text, on_sentence=None, context=None):
            if on_sentence: on_sentence('New Delhi.')
            return 'New Delhi.'
    ai.engine = Conversation()
    try:
        kernel.event_bus.emit('user_message', GOAL)
        assert planning.wait(2)
        kernel.event_bus.emit('user_message', 'What is the capital of India?')
        assert 'New Delhi.' in messages
        assert coordinator.busy
        release.set(); coordinator.queue.join()
        assert coordinator.snapshot()['phase'] == 'awaiting_approval'
        assert len(kernel.approval_manager.list_pending()) == 1
        kernel.event_bus.emit('user_message', 'What is the capital of India?')
        assert len(kernel.approval_manager.list_pending()) == 1
        kernel.event_bus.emit('user_message', 'approve coding step')
        coordinator.queue.join()
        assert coordinator.snapshot()['phase'] == 'completed'
        task = kernel.task_manager.get(coordinator.snapshot()['task_id'])
        assert task.status.value == 'completed'
        assert workspace.joinpath('main.py').read_text() == 'print("ASTA_OK")\n'
    finally:
        release.set(); coordinator.shutdown(); _shutdown(tasks, ai, tools); kernel.shutdown()


def test_configure_services_preserves_unrelated_env_and_removes_duplicate_keys(tmp_path):
    from scripts.configure_model_services import update_env
    env = tmp_path/'.env'
    env.write_text('PRIVATE_KEY=secret\nASTA_MODEL_MODE=old\nASTA_MODEL_MODE=duplicate\n# note\n')
    update_env(env, {'ASTA_MODEL_MODE':'dual','ASTA_BACKGROUND_TASKS':'1'})
    assert env.read_text() == 'PRIVATE_KEY=secret\nASTA_MODEL_MODE=dual\n# note\nASTA_BACKGROUND_TASKS=1\n'


def test_download_services_uses_pinned_manifest_without_downloading(tmp_path):
    import re
    from scripts.download_model_services import download_models
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path/kwargs['filename'])
    paths = download_models(tmp_path, download=download)
    assert len(paths) == 2
    assert calls[0]['filename'] == 'LFM2.5-2.6B-Q4_K_M.gguf'
    assert calls[1]['filename'] == 'Qwen3.5-4B-Q4_K_M.gguf'
    assert all(re.fullmatch(r'[0-9a-f]{40}', item['revision']) for item in calls)
    assert all(item['local_dir'] == str(tmp_path.resolve()) for item in calls)
