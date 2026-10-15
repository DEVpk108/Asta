# Native post-response screenshot sequencing

Base: `origin/refactor/native-runtime-context` (`6c7599f`). This slice replaces
`ai/final_runtime_patch.py` with an inert compatibility entry point and removes
its startup call. The remaining AI runtime hook delegates mixed requests to
native handling; it no longer contains an immediate response-then-capture path.

## Behavior

- A conversational request such as “Tell me a joke then take a screenshot”
  builds the existing low-risk screenshot request through ToolRequestBuilder.
- Native consent and background-control handling run first. A router-owned
  compound command sequence remains owned by the normal planner.
- `PostResponseCapture` creates a random response ticket before generation.
  Ordinary sentence/response event signatures and streamed speech are unchanged.
- Once a nonempty, non-tool-markup response completes, Python publishes its
  final filtered text and ticket over the authenticated HUD transport.
- The renderer replaces this ticket's streamed assistant fragments with the
  completed response without a fade-in animation, switches to the conversation
  view, and acknowledges the
  same ticket after two animation-frame callbacks, only while the document is
  visible. Older messages and user messages are not removed.
- A matching acknowledgement arms a 200 ms settle delay. An unavailable,
  disconnected, old, hidden or unresponsive HUD instead triggers a fallback
  after 1.5 seconds, followed by the same 200 ms settle delay.
- Acknowledgement and timeout race through a single state transition: exactly
  one screenshot request is dispatched. Normal ToolRuntime policy still applies.
- Request metadata distinguishes `hud_acknowledged` from `timeout_fallback`
  and records `capture_response_id`. A fallback is **not** render evidence.
- New user turns, speech interruption and shutdown cancel pending work.
  Empty responses, generation exceptions and model tool-call markup do not
  schedule a screenshot. Shutdown waits for already claimed tool dispatches
  to finish; reentrant shutdown does not wait on its own callback.
- Render tickets are ephemeral and never replayed after reconnect. Python
  validates acknowledgement shape and the capture coordinator matches the
  current pending ticket. Electron also checks the originating webContents.

## Scope and limitations

This is a response ordering/cleanup fix, not an asynchronous EventBus rewrite.
The paint acknowledgement is not semantic visual verification: it cannot prove
that the window is unoccluded, foreground, or visible in the eventual screenshot.
A minimized HUD uses the bounded fallback. The desktop capture backend is
unchanged. Full responses remain displayed using the existing safe Markdown
subset; only the current live rendering is coalesced, not SQLite history.

The remaining runtime, creator-identity, voice and presentation hooks are not
retired by this patch. No dependencies are added. JavaScript protocol tests use
Node's built-in test runner and Python skips that wrapper if Node is absent.

## Validation

New coverage includes deterministic cancelled-timer callbacks, acknowledgement
versus timeout races, stale tokens, generation failure, consent ordering,
compound commands, barge-in, shutdown, legacy-hook delegation, real ToolRuntime
with a fake screenshot backend, HUD subscription cleanup, and renderer/preload/
Electron transport tests with stubs. Tests do not require a running model, audio
hardware or a real Electron window. Perform a live Windows/HUD smoke test after
the Windows pytest suite is green.
