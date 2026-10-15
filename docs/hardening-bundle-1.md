# Hardening bundle 1: native handlers and immediate correctness fixes

Base: published `refactor/post-response-capture`, commit `753ba4f`.

## Implemented together

1. **Remaining AI runtime hook retirement.** `apply_ai_runtime_patch()` is now
   an inert compatibility entry point. Startup no longer calls it, and package
   import no longer replaces `AIModule.on_user_message`.
2. **Native shortcut ownership.** Screenshot/open-screenshot and conversational
   filler handling are owned by native routing after approval/background checks.
   Compound tool sequences and post-response captures keep their existing owners.
   Creator-identity variants work natively and cannot skip consent handling.
3. **Native tool-response policy.** Extracted result suppression and tool wording
   into `ai/response_formatter.py`. Existing AIModule staticmethod APIs delegate
   to it. Intermediate/superseded results remain suppressed.
4. **Strict visual-completion wording.** Both final-result wording and regular
   tool-success wording use the canonical boolean visual-inspection predicate,
   not truthiness of strings or integers.
5. **Native voice cooldown.** The prior two-second wakeword settling period after
   turning off conversation mode now lives in VoiceModule, preserving its other
   listening guards without cross-package monkeypatching.
6. **Background snapshot race.** A cleared snapshot cannot produce the reported
   None-subscript crash in `_start_goal`.
7. **PNG conversion.** Dependency-free extended byte slicing replaces the Python
   per-pixel loop. Tests verify pixel order, scanline boundaries, PNG checksums
   and invalid buffers/dimensions. No NumPy/Pillow dependency is added.
8. **Clipboard race/double-entry protection.** The paste payload is no longer
   restored to the old clipboard immediately after queuing Ctrl+V. Copy failure
   can fall back to direct typing; hotkey failure cannot, because a paste may
   already have been queued.
9. **JSON boundary/repair fixes.** Intent resolution uses `JSONDecoder.raw_decode`
   with absolute offsets rather than a greedy brace-span regex. Truncated-array
   repair preserves the last legitimate string. The unused decision-prompt tool
   definition payload is removed.

## Deliberate clipboard trade-off

Successful clipboard paste leaves the pasted text on the clipboard until the
next copy operation and returns `clipboard_restored: false`. This avoids relying
on a guessed fixed sleep for an asynchronous target application. It does not
claim the target accepted the text; the existing GUI verification layer remains
responsible for verifying effects. Automatic restoration can be reconsidered
only with a reliable consumption/ownership strategy, not another timing guess.

## Remaining planned bundles

- **Bundle 2 — state and lifecycle reliability:** shared active-session ownership,
  memory operation locking and coherent turn storage, then the remaining timer,
  recovery/cache, GUI serialization and shutdown findings after confirming them.
- **Bundle 3 — maintenance and operational cleanup:** wakeword data contracts,
  repository/dependency hygiene, logging/configuration, and additional focused
  handler decomposition. Changes must preserve optional backend behavior.

This bundle is not a claim that every review finding is fixed. No repo-history
rewrite, biometric-data migration, dependency replacement, asynchronous-bus
rewrite or new plugin/autonomy capability is included. Large new capabilities
are separate feature work. Dormant compatibility files are retained for callers;
no active AI/Voice class-replacement hooks remain in the audited packages/startup.
AIModule still needs further decomposition despite extracting the formatter.

## Validation and handoff

55 new Python regressions cover native-hook idempotency, creator consent ordering,
shortcut dispatch, conversation mode, strict visual verdicts, task-result
suppression, JSON boundaries/repair, PNG parity, clipboard sequencing, voice
cooldown and snapshot races. Voice adapter methods are compiled from the actual
source for tests that do not have optional audio dependencies installed.

Run the focused suite and the complete Windows suite before publishing this
branch. Check ordinary screenshot/open-screenshot, creator identity, conversation
mode on/off, post-response capture and interruption in the real HUD/voice runtime.
Unit tests and synthetic PNG benchmarks are not live Windows hardware evidence.
