# Current semantic inspection as visual-goal evidence

The cognitive completion gate previously scanned all historical observations and
accepted any truthy `verified`, `visual_match`, `window_detected`, or `visible`
flag, regardless of tool source or age. Merging nested model analysis also let a
raw positive override VisionInspectTool's final negative confidence/title guard.

## New contract

- Completion requires a successful current `vision.inspect` result with the
  canonical top-level Boolean `verified=True`. A supplied `visual_match` must
  also be Boolean true. Visibility alone, raw model analysis, string values and
  integer truthiness do not authorize visual completion.
- The newest observation must be the same successful semantic inspection, with
  equal output and matching request/plan-step IDs when supplied. Older positives
  cannot be revived after another tool result or a malformed latest observation.
- If action history is supplied, its latest tool must also be the successful
  semantic inspector. Any intervening tool is conservatively an invalidation,
  including read-only tools; this may require another inspection.
- Runtime observation and decision payloads carry request/plan-step identity.
- The legacy `visual_verification` / `vision.bonsai` observation shape is retained
  only when bound to a current successful `vision.inspect` result with matching
  output and identity. Arbitrary `vision.*` sources are not accepted.
- The runtime fast-completion and unconfirmed-vision paths use the same canonical
  Boolean verdict as the cognitive gate. They do not reinterpret nested analysis.

Non-visual goals, tool approvals, grounding/click selection and the inspector's
existing confidence/title policy are unchanged. The patch trusts the registered
semantic inspector's verdict; it does not prove that the visual model is correct,
check arbitrary prompts against the goal, secure a malicious tool registry,
deduplicate externally replayed results, or lock the desktop against external
changes after capture. No timestamp-based wall-clock freshness promise is made.

## Local regression command

```sh
python -m pytest -q tests/test_visual_verification_provenance.py tests/test_agent_brain.py tests/test_agent_visual_grounding.py tests/test_cognitive_planner.py tests/test_mixed_task_runtime.py tests/test_task_manager_lock_order.py tests/test_readback_before_execution.py tests/test_verification_recovery_reentry.py
```

Tests include a real AgentBrain with an inert completion-claiming provider, the
real VisionInspectTool with injected capture/model stubs on either side of its
confidence threshold, and real runtime follow-up generation after a click.
No model server, microphone, Docker daemon or desktop control is required.

## Validation for this patch

- Exact handoff command: **128 passed**.
- Combined cognitive/concurrency and previous safety regressions: **327 passed**.
  That expanded set uses a dummy Spotify client ID for legacy API-path fixtures,
  not live authentication or Spotify access.
- New provenance tests: **48 passed**. Before the fix, 45 failed and 3 passed.
- Full Linux/Python 3.13 suite: **891 passed, 61 failed, 10 collection errors**.
  Failure/error IDs are identical to the parent lock-order patch (843 passed).
- A wider targeted selection that included optional-audio and vision-engine
  fixtures has 48 failures; all are in the parent full-suite failure ID set.
  They are not repaired by this patch, and the full suite is not green.
- Compilation, whitespace and patch-application checks passed. Windows execution
  and real screenshot/model/hardware validation remain local follow-ups.
