# Hardening Bundle 2 — session ownership, memory, and lifecycle

Base: Hardening Bundle 1 (`fix/hardening-bundle-1`). This is the second of the
three bundled review-fix batches, not a new capability release.

## Changes

- **One chat store and active session:** Kernel owns the shared SQLite
  `ChatHistoryStore`; AI and HUD initialize it idempotently and do not close it
  independently. Store methods serialize connection lifetime and session changes.
  AI context reads the selected session, not whichever session most recently
  received a message. Selecting, creating, or deleting the active chat emits a
  session-change event that cancels pending capture and resets model context.
  Old-generation chunks/finals are dropped after a session change, even if the
  user switches away and back to the same chat.
- **Long-term memory:** user messages and full generated conversational replies
  are stored, rather than every streamed sentence/status notification. A dedicated
  `conversation_turn_completed(text, session_id)` event supplies the reply's
  originating session. Queue entries capture their session at enqueue time.
  Module and adapter locks serialize writes, recall, deletion and close. Deleted
  session IDs are tombstoned for the module lifetime so queued writes cannot
  recreate their memories. Shutdown stops new writes and drains accepted writes;
  if a write exceeds the bounded join, the writer closes the backend on exit.
  Memory recalls remain cross-session long-term recall; this is not a new
  per-session privacy/access-control system.
- **Approval races:** pending approvals are consumed atomically, including late
  approval/expiry races. Duplicate delivery is a no-op, not another failure card.
  Expiry failures retain task metadata. Timers are tracked, identity checked and
  cancelled on reply/shutdown. Sandbox-approved retry entries are capped at 64.
  An already claimed/in-flight tool is not retroactively cancelled by shutdown.
- **Model lock order:** routed clients do not hold the client/history lock while
  waiting at runtime/profile admission. Stateful generation still serializes its
  history updates. Local llama.cpp request/history entry points use a re-entrant
  request lock; cancellation remains callable without acquiring it.
- **Desktop conflicts:** the shared dispatcher serializes GUI-affecting tool
  calls through a kernel-owned desktop lock, after authority checks. This covers
  computer/vision tools, project program launches, media UI/control, browser
  search, and application open/launch/close. Non-GUI work remains independent.
  This is per-tool serialization, not a transaction across an entire GUI plan.
- **Resource lifecycle:** Spotify playback uses the existing 401-refresh helper;
  its setup browser closes in `finally`. Sandbox pre-run helper probes are safe.
  Project launches use an anonymous temporary stderr file whose parent handle is
  closed on every exit, rather than leaving named log files behind. Speech closes
  its stream only after the worker exits; a stuck worker defers close to its own
  `finally` rather than closing a device underneath an active write.
- **Bounded caches:** song lookup cache is lock protected, copy-on-read, and LRU
  capped at 256. Task creation prunes terminal history to the latest 100 terminal
  tasks by creation order; paused/unfinished tasks are retained. The terminal cap
  is enforced on the next create, not by deleting active work on completion.

Existing `assistant_sentence` and `assistant_response` payloads are unchanged.
Three existing test fixtures now reflect shared session ownership, selected-chat
reads, and idempotent duplicate approvals; their original behavioral assertions
remain, except the deliberately removed duplicate-failure notification.

## Validation and scope

`tests/test_hardening_bundle_2.py` adds 50 deterministic tests with fake models,
backend, timers, browser/launch adapters and no live credentials or services.
Speech lifecycle tests compile the real class without its optional audio import;
they verify ownership, not PortAudio/device behavior.

Local combined focused suite: 490 passed. JavaScript: 11 test-runner entries
passed (capture protocol plus avatar files). New regression file passed in five
consecutive runs.

Full local Linux suite: 1078 passed, 59 failed, 10 collection errors. The failure
and error IDs are identical to the Bundle 1 baseline (1028 passed, 59 failed,
10 errors): optional audio dependencies, Windows-specific behaviors and existing
service fixtures are not available/green in this environment. This is baseline
parity, **not** a full-suite pass. Run the complete suite on the Windows dev setup.

After tests, smoke-check two chat sessions (switch back to the older one and ask
what was discussed), a fresh conversation, approval/rejection/expiry, one desktop
program launch, voice interrupt and normal shutdown. No automated test here
proves live GPU/audio/browser/device performance.

## Remaining batch

Bundle 3 is the repository/data/configuration cleanup and remaining confirmed
small fixes: wakeword sample-rate/data contracts, dead/duplicate artifacts,
packaging/docs and targeted operational cleanup after checking actual call sites.
No Git history rewrite, deletion of user data, plugin autodiscovery, asynchronous
bus migration, or new autonomy capability is included in this patch. Larger
feature additions and further AI-module decomposition require their own scoped
work; completing these batches must not be described as implementing the entire
long-term roadmap.
