# Consolidation slice 1: retire the memory pass-through wrapper

Base: published `safety-baseline-2026-10` (`d7f5bcd`). The base tree is identical
to the previously reviewed visual-verification checkpoint; the user's complete
Windows baseline passed 1,040 tests before this refactor.

## What changed

- Main no longer imports or calls `apply_memory_runtime_patch()`.
- The old callable remains as a compatibility no-op. It no longer imports
  AIModule, captures a generation method, replaces it, or keeps patch state.
- Its documentation now reflects the actual ownership boundaries: MemoryModule
  recalls into `kernel.memory_context`, and ContextBuilder decides whether that
  recall belongs in a turn's runtime prompt.

The former wrapper only called the existing generation method with the same
text and runtime_context and returned its result. Removing it does not remove
memory, add new prompt content, change memory relevance rules, or change writes.

## Deliberately unchanged

- MemoryModule's recall-before-write order and background writer lifecycle.
- Native generation, sentence streaming and completed-response events.
- Explicit caller context and the retained context enrichment patch.
- Main's module registration order, including memory before AI.
- MemPalace locking, sentence-based indexing and chat-session ownership.
- The active AI, final and context runtime hooks, plus AI's import-time creator
  wrapper. These still need separate consolidation; this is not the whole refactor.

The compatibility file is intentionally retained for older launchers. It is no
longer an active monkeypatch, but deleting compatibility entry points is a
separate decision. No user memory files or chat database are migrated.

## Tests

```sh
python -m pytest -q tests/test_memory_runtime_retirement.py tests/test_context_builder.py tests/test_context_runtime.py tests/test_approval_consent_safety.py tests/test_readback_before_execution.py tests/test_verification_recovery_reentry.py tests/test_visual_verification_provenance.py
```

The 18 new tests cover method identity, repeated legacy calls, exact argument
and return forwarding, a later method replacement, startup wiring, the
compatibility file's lack of an AI import, and recall/context/streaming/write
behavior with and without a legacy caller. The memory backend and LLM are inert
fixtures; no live MemPalace, embedding/model server or audio device is needed.
Global method and patch flags are restored to prevent test-order pollution.

## Validation

- Exact handoff command: **197 passed**.
- New tests: **18 passed**; pre-retirement characterization: 9 failed, 9 passed.
- Full Linux/Python 3.13: **909 passed, 61 failed, 10 collection errors**.
  The failing/error IDs match the published-base comparison exactly (891 passed);
  the 18 new tests account for the increase. The Linux suite is not fully green.
- Python compilation, whitespace and application checks passed.
- Windows regression and live memory smoke testing remain local follow-ups.

After focused tests pass locally, rerun the full Windows suite before migrating
another active runtime patch. The published safety tag remains unchanged.
