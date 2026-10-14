# Consolidation slice 2: native per-turn context preparation

Base: the memory-wrapper retirement, validated locally by the user with
1,058 passing Windows tests. This slice supersedes the context wrapper retained
in slice 1; the published safety tag is unchanged.

## Ownership

AIModule._prepare_runtime_context now owns the policy previously implemented by
ai/context_runtime_patch.py. Native _generate_response calls it before the model.
Main no longer imports or invokes apply_context_runtime_patch. The callable is
retained as a compatibility no-op without patch flags or class mutation.

The generation implementation inside apply_ai_runtime_patch was structurally
identical to the native implementation (AST bodies matched, excluding its
docstring). That duplicate and its method assignment are removed, so applying
the remaining AI hook cannot overwrite the native context-aware generation.
Its other AI/voice behavior remains in that hook and is not consolidated here.

## Preserved production semantics

- Existing kernel.context_builder is reused; a missing one is created and cached.
- Truthy explicit context is forwarded unchanged, without intent routing or
  prompt rebuilding. As before, a missing builder is created before that check.
- Falsy context (including an empty string) follows the existing rebuild policy:
  analyze the request, then call build_prompt once for that generation turn.
- Router/build_prompt exceptions log the existing context error and continue
  generation with context=None. Builder construction remains outside that
  fallback region, as it was in the old wrapper; this is not a new error policy.
- Model errors are not swallowed as context failures.
- Sentence filtering/streaming, completed-response events, tool-markup handling,
  empty responses, memory relevance and explicit caller context are unchanged.

Direct native generation now gets the same enrichment as production startup;
callers no longer need to install a wrapper first. There is no additional recall
or memory write. Intent analysis already performed elsewhere is not reused in
this slice: changing that would be a separate behavior/performance change.

Two explicit startup hooks remain (AI and final), as does the AI package's
import-time creator wrapper. This is not the complete AIModule refactor.

## Regression command

```sh
python -m pytest -q tests/test_native_runtime_context.py tests/test_memory_runtime_retirement.py tests/test_context_runtime.py tests/test_context_builder.py tests/test_model_tool_calls.py tests/test_approval_consent_safety.py tests/test_readback_before_execution.py tests/test_verification_recovery_reentry.py tests/test_visual_verification_provenance.py
```

New tests use inert engines/builders and execute the real AI compatibility hook
with only its optional VoiceModule dependency substituted. They test both legacy
hook orders, repeated application, method ownership, explicit/falsy contexts,
builder reuse, routing/build failure, streaming, empty responses and model errors.
The combined set also exercises real ContextBuilder/MemoryModule pipeline fixtures.
No live model, microphone, embedding backend, Docker or desktop action is required.

## Validation

- Exact handoff command: **228 passed**, including **24 new tests**.
- Pre-migration native-context regressions: 18 failed, 6 passed.
- Linux parent: **909 passed, 61 failed, 10 collection errors**.
- Linux after migration: **933 passed, 61 failed, 10 collection errors**.
  New/removed failure or error IDs: none. Optional audio/platform/model lifecycle
  failures remain; this does not claim a green Linux suite.
- Compilation, whitespace and patch-application checks passed.
- Full Windows suite must pass again before the next consolidation slice.
