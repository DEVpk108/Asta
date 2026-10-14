# Verified readback before project execution

Every planned file write must have a matching, verified file readback before
`system.run_command`, `project.open_program`, or another write to the same file.
The execution barrier applies even when execution names another file: imports
and configuration can depend on any earlier write.

## Two enforcement layers

1. Planner normalization only reuses an existing readback when it is before
   execution and the next write to that file. Otherwise it inserts a readback
   directly after the write, copying the expected path/content from the write.
   Original subsequent steps are preserved; SHA-256 step references are shifted
   when an insertion changes positions.
2. The independent plan validator rejects a matching readback that occurs too
   late. It does not rely on normalization having run. Each successive write
   requires fresh readback, even when the content is unchanged.

The task/tool runtimes still perform the actual file verification and enforce
execution approval. A readback mismatch does not authorize the following run.
Planning never reads or writes the user's files merely to repair the step order.

Malformed supplementary readbacks still fail their existing validation checks;
the ordering fix does not bypass path/content validation or invent capabilities.
This change does not lock files against external modification after verification,
sandbox host code, or fix the separately tracked verification-recovery re-entry.

## Regression checks

```sh
python -m pytest -q tests/test_readback_before_execution.py tests/test_project_planner_contract_fix.py tests/test_project_edit_fix_loop.py tests/test_desktop_program.py tests/test_workspace_project_task.py tests/test_mixed_task_runtime.py tests/test_concurrent_models.py
```

Tests cover sandbox and desktop execution, action-only proposals, multiple files,
repeated writes, Windows path separators, hash-reference remapping, input
immutability, idempotent normalization, and independent validator rejection.
Runtime tests use real workspace/approval/verification code with an inert
container fixture; they verify that readback precedes execution approval and a
failed readback prevents execution. No live model, Docker daemon or desktop
program is needed.
## Validation for this patch

- Focused project, runtime, concurrent-model and approval regressions: **203 passed**.
- Full Linux/Python 3.13 run: **814 passed, 62 failed, 10 collection errors**.
  The 72 failing/error test IDs are identical to the preceding approval-safety
  baseline (793 passed); this patch adds 21 passing tests and no new failure IDs.
  This is not a fully green cross-platform suite or a Windows hardware test.
- Python compilation and `git diff --check` passed.
