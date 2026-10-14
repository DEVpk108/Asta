# Task-manager preemption notifications outside the mutation lock

TaskManager previously emitted `task_paused` synchronously from
`_activate_locked`. BackgroundTaskCoordinator's subscriber acquires the
coordinator lock, while its `start()` path holds that lock and queries
TaskManager. Two threads could therefore wait on each other's locks (AB-BA).

## Change

- The locked helper commits task states and active-task identity, then returns
  the previous task's paused snapshot instead of invoking subscribers.
- `create`, `activate`, and `resume` publish any deferred pause after releasing
  the manager lock, before their existing creation/activation/resume events.
- Pause subscribers see the committed replacement task through `current()`;
  the event payload still identifies the previous task and its paused state.
- A completed, failed or cancelled target is rejected before pausing the valid
  current task. Previously the rejected switch could leave that task paused.
- Inactive creation and reactivating the current task do not emit a preemption.

This removes the identified manager-to-coordinator callback lock inversion.
It does not introduce asynchronous events or guarantee total event ordering
across threads. Payloads describe a captured transition, not a promise that no
subsequent transition has occurred. Unrelated coordinator locking, the separate
background-state snapshot race, and TaskManager history retention are unchanged.

## Regression command

```sh
python -m pytest -q tests/test_task_manager_lock_order.py tests/test_task_manager.py tests/test_concurrent_models.py tests/test_mixed_task_runtime.py tests/test_verification_recovery_reentry.py tests/test_readback_before_execution.py
```

The new tests cover all three preemption entry points, event ordering, querying
from a subscriber's second thread, invalid terminal targets, and a coordinated
interleaving of the real BackgroundTaskCoordinator.start/on_terminal paths.
A test-only timed lock makes the broken implementation fail rather than leave
permanently deadlocked test threads. No sleep-based interleaving, live model,
microphone, Docker daemon or desktop launch is required.

## Validation for this patch

- Exact handoff command: **83 passed**.
- Combined concurrency, recovery and approval/readback regressions: **246 passed**.
- New concurrency suite: **16 tests passed**, repeated 20 times (**320 test
  executions**) without failures. Before the fix, 13 failed and 3 passed.
- Full Linux/Python 3.13 suite: **843 passed, 61 failed, 10 collection errors**.
  The failure/error test IDs are identical to the parent recovery patch
  (827 passed). This patch adds 16 passing tests; it does not fix the remaining
  suite failures or constitute Windows hardware validation.
- Compilation, whitespace and application checks passed.
