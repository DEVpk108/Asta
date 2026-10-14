"""Preemption callbacks must not run inside TaskManager's mutation lock."""
import threading

import pytest

from core import EventBus, Kernel, TaskManager
from core.background_tasks import BackgroundTaskCoordinator
from core.contracts import TaskStatus


def _switch(manager, action, target):
    if action == 'create':
        return manager.create('replacement')
    return getattr(manager, action)(target.id)


@pytest.mark.parametrize('action', ['create', 'activate', 'resume'])
def test_preemption_subscriber_can_query_manager_from_another_thread(action):
    bus = EventBus()
    manager = TaskManager(event_bus=bus)
    previous = manager.create('previous')
    target = manager.create('replacement', activate=False)
    if action == 'resume':
        manager.activate(target.id)
        manager.activate(previous.id)
    probes, completed, observed = [], [], []

    def paused(task):
        done = threading.Event()

        def query():
            observed.append(manager.current())
            done.set()

        probe = threading.Thread(target=query, daemon=True)
        probes.append(probe)
        probe.start()
        completed.append(done.wait(1))

    bus.subscribe('task_paused', paused)
    replacement = _switch(manager, action, target)
    for probe in probes:
        probe.join(2)
    assert completed == [True]
    assert observed == [replacement]
    assert previous.status is TaskStatus.PAUSED
    assert manager.current() is replacement


@pytest.mark.parametrize('action', ['create', 'activate', 'resume'])
def test_preemption_event_order_and_committed_state(action):
    bus = EventBus()
    manager = TaskManager(event_bus=bus)
    previous = manager.create('previous')
    target = manager.create('replacement', activate=False)
    if action == 'resume':
        manager.activate(target.id)
        manager.activate(previous.id)
    events = []
    for name in ['task_paused', 'task_created', 'task_activated', 'task_resumed']:
        bus.subscribe(name, lambda task, name=name:
            events.append((name, task, manager.current())))
    replacement = _switch(manager, action, target)
    expected = {'create': ['task_paused', 'task_created', 'task_activated'],
                'activate': ['task_paused', 'task_activated'],
                'resume': ['task_paused', 'task_resumed']}[action]
    assert [event[0] for event in events] == expected
    assert events[0][1]['id'] == previous.id
    assert events[0][1]['status'] == 'paused'
    assert all(event[2] is replacement for event in events)
    assert all(event[1]['id'] == replacement.id for event in events[1:])


def test_inactive_creation_does_not_pause_current_task():
    bus = EventBus()
    manager = TaskManager(event_bus=bus)
    previous = manager.create('previous')
    pauses = []
    bus.subscribe('task_paused', lambda task: pauses.append(task))
    target = manager.create('pending', activate=False)
    assert manager.current() is previous
    assert target.status is TaskStatus.PENDING
    assert pauses == []


@pytest.mark.parametrize('action', ['activate', 'resume'])
def test_reactivating_current_task_does_not_emit_preemption(action):
    bus = EventBus()
    manager = TaskManager(event_bus=bus)
    current = manager.create('current')
    pauses = []
    bus.subscribe('task_paused', lambda task: pauses.append(task))
    assert getattr(manager, action)(current.id) is current
    assert current.status is TaskStatus.ACTIVE and pauses == []


@pytest.mark.parametrize('action', ['activate', 'resume'])
@pytest.mark.parametrize('terminal', ['complete', 'fail', 'cancel'])
def test_invalid_terminal_target_does_not_pause_active_task(action, terminal):
    bus = EventBus()
    manager = TaskManager(event_bus=bus)
    previous = manager.create('previous')
    target = manager.create('terminal target', activate=False)
    if terminal == 'fail':
        manager.fail('fixture failure', target.id)
    elif terminal == 'cancel':
        manager.cancel('fixture cancellation', target.id)
    else:
        manager.complete(task_id=target.id)
    events = []
    bus.subscribe('task_paused', lambda task: events.append(task))
    with pytest.raises(ValueError, match='cannot activate task'):
        getattr(manager, action)(target.id)
    assert manager.current() is previous
    assert previous.status is TaskStatus.ACTIVE
    assert events == []


class _BoundedLock:
    """Make a regression fail rather than leaving deadlocked test threads."""
    def __init__(self):
        self.lock = threading.RLock()
        self.timeouts = []

    def __enter__(self):
        if not self.lock.acquire(timeout=2):
            self.timeouts.append(threading.current_thread().name)
            raise TimeoutError('test detected coordinator/task-manager lock inversion')
        return self

    def __exit__(self, *args):
        self.lock.release()


def test_real_background_start_and_preemption_avoid_abba_lock_inversion():
    kernel = Kernel()
    manager = kernel.task_manager
    previous = manager.create('background project')
    target = manager.create('replacement', activate=False)
    lock_held, preempting = threading.Event(), threading.Event()
    # Signal before the coordinator's real synchronous on_terminal subscriber.
    kernel.event_bus.subscribe('task_paused', lambda task: preempting.set())
    coordinator = BackgroundTaskCoordinator(kernel)
    coordinator.lock = _BoundedLock()
    coordinator.state = {'job_id': 'fixture', 'task_id': previous.id,
                         'phase': 'working', 'goal': previous.goal, 'message': ''}
    outcomes, errors = [], []

    def start_under_coordinator_lock():
        try:
            with coordinator.lock:
                lock_held.set()
                if not preempting.wait(2):
                    raise TimeoutError('preemption was not reached')
                # Real start() acquires the task-manager lock through current().
                outcomes.append(coordinator.start('another project', None))
        except Exception as exc:
            errors.append(exc)

    def activate():
        try:
            manager.activate(target.id)
        except Exception as exc:
            errors.append(exc)

    starter = threading.Thread(target=start_under_coordinator_lock, daemon=True)
    preemptor = threading.Thread(target=activate, daemon=True)
    try:
        starter.start()
        assert lock_held.wait(2)
        preemptor.start()
        starter.join(4)
        preemptor.join(4)
        assert not starter.is_alive() and not preemptor.is_alive()
        assert errors == [] and coordinator.lock.timeouts == []
        assert outcomes == [False]  # an active replacement must reject new work
        assert manager.current() is target and previous.status is TaskStatus.PAUSED
        assert coordinator.snapshot()['phase'] == 'paused'
    finally:
        coordinator.state = None
        coordinator.shutdown()
        kernel.shutdown()
