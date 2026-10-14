# Single-attempt verification failure recovery

A successful tool execution with a failed postcondition is one attempt, not two
executions. The runtime now routes that failure directly into a shared recovery
helper instead of emitting a synthetic `tool_result` on the global event bus.

## Evidence and recovery contract

- The one evidence entry preserves the real tool's success, output and request
  ID, alongside the failed verification and recovery decision. Tool success
  remains distinct from a verified outcome.
- Recovery receives a private failed-result view with the original output,
  duration and metadata for diagnosis. The original published result is not
  mutated or republished.
- Recovery decides before journaling the current attempt, so history contains
  only prior failures. The current attempt is recorded before dispatching a real
  retry or autonomous replan.
- Retry budget, approval, sandbox repair, and deterministic command behavior are
  unchanged. Commands and desktop launches still go to diagnosis/replanning,
  not a blind identical retry.
- Unknown/unavailable verification still blocks and pauses the task. Terminal
  failures and unavailable diagnosis/replanning paths also retain the current
  evidence entry.

Dedicated task verification/recovery events remain available to UI consumers.
Consumers should not equate raw tool success with successful task verification.
A genuine retry still publishes its own real result; this change does not make
EventBus asynchronous, deduplicate repeated externally delivered results, or
resolve unrelated lock-order or approval-timer races.

## Local regression command

```sh
python -m pytest -q tests/test_verification_recovery_reentry.py tests/test_autonomy_verification.py tests/test_recovery.py tests/test_readback_before_execution.py tests/test_project_edit_fix_loop.py tests/test_workspace_project_task.py tests/test_sandbox_self_repair.py
```

The fixtures exercise real task/tool/recovery code with inert tools and mock
observers. No microphone, model server, Docker daemon or desktop launch is used.

## Validation for this patch

- Exact handoff command above: **92 passed** (no special environment setting).
- Expanded focused suite: **268 passed**, including prior approval/readback,
  diagnosis, replanning and sandbox-repair regressions. This suite sets a dummy
  `ASTA_SPOTIFY_CLIENT_ID=test-client` for legacy API-path fixtures; it does not
  authenticate or contact Spotify.
- Full Linux/Python 3.13 run: **827 passed, 61 failed, 10 collection errors**.
  Relative to the preceding readback patch (814 passed, 62 failed, 10 errors),
  12 new tests pass and the existing playback-verification retry regression is
  fixed. No new failing/error test IDs. Remaining failures are not repaired here.
- An isolated legacy media-sequence test without the dummy client setting fails
  on both the parent and this patch; this is distinct from the fixed verification
  retry regression.
- Compilation, whitespace and patch-application checks passed. Windows execution
  and hardware validation remain to be run locally.
