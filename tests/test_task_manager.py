from core import EventBus, Kernel, TaskManager
from core.contracts import TaskStatus


def test_create_and_activate_task():
    events = []
    bus = EventBus()
    bus.subscribe("task_created", lambda **payload: events.append(("created", payload["task"])))
    bus.subscribe("task_activated", lambda **payload: events.append(("activated", payload["task"])))

    manager = TaskManager(event_bus=bus)
    task = manager.create(
        "Integrate the new A.S.T.A. agent runtime",
        constraints=["keep main stable"],
        pending_steps=["define task contract", "add manager", "connect runtime"],
    )

    assert task.status is TaskStatus.ACTIVE
    assert manager.current() is task
    assert task.goal == "Integrate the new A.S.T.A. agent runtime"
    assert task.constraints == ["keep main stable"]
    assert events[0][0] == "created"
    assert events[1][0] == "activated"


def test_switching_tasks_pauses_previous_task():
    events = []
    bus = EventBus()
    bus.subscribe("task_paused", lambda **payload: events.append(payload["task"]))

    manager = TaskManager(event_bus=bus)
    first = manager.create("First task")
    second = manager.create("Second task")

    assert first.status is TaskStatus.PAUSED
    assert second.status is TaskStatus.ACTIVE
    assert manager.current() is second
    assert events[-1]["id"] == first.id
    assert events[-1]["status"] == TaskStatus.PAUSED.value


def test_progress_and_evidence_are_recorded():
    manager = TaskManager()
    task = manager.create(
        "Improve the voice pipeline",
        pending_steps=["inspect VAD", "tune thresholds"],
    )

    manager.set_step("inspect VAD")
    manager.complete_step("inspect VAD")
    manager.add_evidence({"source": "test", "result": "VAD file inspected"})

    assert task.current_step is None
    assert task.completed_steps == ["inspect VAD"]
    assert task.pending_steps == ["tune thresholds"]
    assert task.evidence[0]["result"] == "VAD file inspected"


def test_terminal_states_clear_the_active_task():
    manager = TaskManager()
    completed = manager.create("Complete this task")
    manager.complete(result={"commit": "abc123"})

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result == {"commit": "abc123"}
    assert manager.current() is None

    failed = manager.create("Another task")
    manager.fail("tool execution failed")

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "tool execution failed"
    assert manager.current() is None


def test_pause_and_resume():
    manager = TaskManager()
    task = manager.create("Pause and resume me")

    manager.pause()
    assert task.status is TaskStatus.PAUSED
    assert manager.current() is None

    manager.resume(task.id)
    assert task.status is TaskStatus.ACTIVE
    assert manager.current() is task


def test_kernel_owns_task_manager():
    kernel = Kernel()
    task = kernel.create_task("Check kernel task integration")

    assert isinstance(kernel.task_manager, TaskManager)
    assert kernel.current_task is task
    assert task.status is TaskStatus.ACTIVE


def test_snapshot_is_json_friendly():
    manager = TaskManager()
    task = manager.create("Create a task snapshot", metadata={"source": "test"})
    snapshot = task.to_dict()

    assert snapshot["id"] == task.id
    assert snapshot["status"] == TaskStatus.ACTIVE.value
    assert snapshot["metadata"] == {"source": "test"}
    assert isinstance(snapshot["created_at"], str)
    assert isinstance(snapshot["updated_at"], str)


def main():
    tests = [
        test_create_and_activate_task,
        test_switching_tasks_pauses_previous_task,
        test_progress_and_evidence_are_recorded,
        test_terminal_states_clear_the_active_task,
        test_pause_and_resume,
        test_kernel_owns_task_manager,
        test_snapshot_is_json_friendly,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - simple standalone runner
            failures += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")

    print("ALL PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_task_can_own_a_structured_plan():
    from core.contracts import Plan, PlanStatus, PlanStep

    manager = TaskManager()
    plan = Plan(
        goal="Open and close an application",
        steps=[
            PlanStep(id="step-1", description="open app"),
            PlanStep(id="step-2", description="close app", depends_on=["step-1"]),
        ],
        status=PlanStatus.READY,
    )

    task = manager.create(
        "Open and close an application",
        plan=plan,
        pending_steps=[step.description for step in plan.steps],
    )

    assert task.plan is plan
    assert task.pending_steps == ["open app", "close app"]
    snapshot = task.to_dict()
    assert snapshot["plan"]["status"] == "ready"
    assert snapshot["plan"]["steps"][1]["depends_on"] == ["step-1"]
