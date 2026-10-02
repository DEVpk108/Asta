from core.autonomy.capability_setup import (
    CapabilitySetupManager,
    CapabilitySetupStatus,
)
from core.task_manager import TaskManager


class FakeMediaManager:
    def __init__(self):
        self.refresh_calls = 0

    def refresh_configuration(self):
        self.refresh_calls += 1


class FakeOperatorResult:
    status = CapabilitySetupStatus.COMPLETED
    summary = "Spotify setup complete."
    redirect_uri = "http://127.0.0.1:8765/callback"
    metadata = {}


def test_capability_setup_manager_runs_operator_and_emits_completion():
    media = FakeMediaManager()
    events = []
    manager = CapabilitySetupManager(
        media,
        event_bus=type(
            "Bus",
            (),
            {"emit": lambda self, name, **payload: events.append((name, payload))},
        )(),
        operators={"spotify": lambda: FakeOperatorResult()},
    )
    tasks = TaskManager()
    task = tasks.create("play hanuman chalisa on spotify", activate=False)

    assert manager.start(task, capability="spotify", step_id="step-2") is True

    import time
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if any(name == "capability_setup_completed" for name, _ in events):
            break
        time.sleep(0.01)

    assert media.refresh_calls == 1
    completed = [
        payload["event"]
        for name, payload in events
        if name == "capability_setup_completed"
    ]
    assert completed
    assert completed[-1]["status"] == "completed"
    assert completed[-1]["step_id"] == "step-2"


def test_capability_setup_manager_deduplicates_inflight_setup():
    media = FakeMediaManager()
    events = []
    entered = False

    def operator():
        nonlocal entered
        entered = True
        import time
        time.sleep(0.05)
        return FakeOperatorResult()

    manager = CapabilitySetupManager(
        media,
        event_bus=type(
            "Bus",
            (),
            {"emit": lambda self, name, **payload: events.append((name, payload))},
        )(),
        operators={"spotify": operator},
    )
    tasks = TaskManager()
    task = tasks.create("play hanuman chalisa on spotify", activate=False)

    assert manager.start(task, capability="spotify", step_id="step-2") is True
    assert manager.start(task, capability="spotify", step_id="step-2") is True

    import time
    time.sleep(0.2)

    assert entered is True
    assert len(
        [name for name, _ in events if name == "capability_setup_started"]
    ) == 1
