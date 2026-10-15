# Review-fix closure matrix

Evidence: the reviewed source, the safety/refactor patches and the three hardening
bundles. "Implemented" means code plus regression coverage; Windows confirmation
for Bundle 3 remains pending until its test output is recorded. Earlier Windows
full-suite gates were 1040 (safety), 1120 (capture), 1175 (Bundle 1), and 1225
(Bundle 2). A passing mock suite is not a production-readiness score.

## Implemented

| Finding | Resolution |
| --- | --- |
| Refusal/filler normalized into approval | One complete-utterance consent policy; explicit approval only; refusal preserved |
| Readback after code execution | Planner inserts/checks write readback before execution barriers |
| Recursive synthetic tool results | Direct verification-failure recovery; no duplicate global tool-result event |
| Task preemption emits under lock | Task snapshots/events published outside manager lock |
| Stale/nonsemantic visual completion | Current semantic inspection provenance and strict boolean flags |
| Shared chat/session ownership | One kernel-owned store; selected-session reads, reset/cancel on switch, stale-generation invalidation |
| Memory write/search/delete/close races | Module/adapter serialization, queue drain before close, deletion tombstones |
| Fragmented assistant memory | Dedicated full conversational-turn completion event; status/sentence fragments excluded |
| Active runtime monkey patches/import mutations | Native context, capture, shortcuts, formatting and voice handling; inert compatibility entry points retained for old imports |
| HUD capture acknowledged too early | Response-correlated HUD acknowledgement, cancellation and bounded fallback |
| PNG per-pixel Python bottleneck | C-backed byte slicing with byte-identical fixture checks (Bundle 1) |
| Clipboard immediate restoration | No early restore; no unsafe double-entry fallback after hotkey failure |
| Intent JSON extraction/truncated arrays | Raw JSON decoding and bounded repairs preserving string/list values |
| Late/duplicate approvals and orphan timers | Atomic take, idempotent duplicate delivery, tracked/cancelled timers, task-aware expiry |
| Model client admission/request races | Client lock released before runtime admission; request/history entry points serialized; cancel remains available |
| Desktop focus conflicts | Shared per-tool desktop dispatch lock after authority checks |
| Speech close while worker active | Worker-owned close after exit; deferred close on bounded join timeout |
| Spotify 401 retry/browser leaks | Shared refresh helper for playback; setup context closes in finally |
| Sandbox probe before initialization | Executable initialized and unavailable checks safe |
| Named temporary program logs | Anonymous temporary stderr file, closed on every parent exit |
| Unbounded song/task terminal caches | Locked LRU song cache; terminal-task retention at creation, unfinished tasks preserved |
| Arithmetic negatives/eval | Signed spoken numbers and allowlisted bounded AST arithmetic |
| WMI polling/dedupe overhead | Short cache, fresh post-action checks, indexed dedupe |
| Windows PID orphan children | Explicit taskkill process-tree invocation through existing authority boundary |
| Repeated workspace persistence | Sanitized-payload change detection, retry after failed write |
| UI search punctuation | Clipboard whole-text fallback instead of interpolated SendKeys text |
| Wakeword sample-rate inconsistencies | Shared prepared-WAV contract, real producer resampling, matching validators, migration warnings |
| Training duplicates/manifest scans | Canonical entry points, installed dependency lookup, membership cache, correct preparer root |
| Unused runtime prototypes | Removed unused Moonshine/Polly/screenshot code; camera experiment isolated and no pickle loading |
| Scattered dependency definitions | Canonical pyproject extras; compatibility requirements entry points |
| Binary-only source documentation | Readable Markdown counterparts added, originals preserved |

## Partial or explicitly not closed

| Area | What remains |
| --- | --- |
| AI-module concentration | Helpers extracted and patches retired, but AIModule is still large; a full orchestration-only decomposition is not complete |
| Presentation/identity configuration | Existing behavior preserved; presentation prose and shortcut phrases are not fully externalized |
| Logging | Main AI/voice/speech/memory/tool-runtime hot paths migrated; other legacy modules/CLI progress still print; no trajectory tracing added |
| Repository privacy/bloat | Ignore rules and audit added; tracked recordings, biometric pickle, original documents and vendored dependency remain pending explicit owner cleanup; historical copies unchanged |
| Wakeword release validation | Existing data must be explicitly migrated/regenerated and models retrained/revalidated; no accuracy or false-positive claim |
| Dependency release hygiene | Unified declarations, not a platform/CUDA lockfile or standalone packaged app |
| UI automation latency | Safe text fallback fixed; persistent PowerShell worker/cold-start consolidation not implemented |
| Abrupt host-crash orphan containers | Normal finally cleanup exists; no broad startup sweep that could kill another active ASTA instance's containers |

## Rechecked findings that do not justify another fix

- The current repository already has a working `.env` loader with environment
  precedence. No new dependency/second loader was needed.
- Current VAD restoration uses the saved `start_chunk_rms`; the review's stale
  version is not a reason to retune working audio thresholds.
- Apparent Devanagari mojibake in an exported code bundle is not evidence of
  corrupt repository source. No blind encoding rewrite was applied.
- Separate chat/planning clients can be appropriate when they share a managed
  model runtime but intentionally isolate conversational state. They should not
  be merged solely because there are two Python provider objects.
- Some PDF paths (such as `core/dialogue/`) are not present in this checkout;
  their specific suggested edits cannot be applied as if those files existed.
- The static avatar shown in the screenshot was reduced-motion behavior, not
  evidence of a broken runtime or an intended exit command.

## Future capabilities, not defects silently included in these patches

Dynamic skill/plugin discovery; typed schemas across every EventBus event;
async/task-queued dispatch migration; calibrated/expanded routing authority;
proactive schedules; richer user profiles; broader multimodal workflows; and
production trajectory evaluation/observability need their own requirements,
safety decisions, implementation and end-to-end gates. The three patch batches
do not claim to implement that entire roadmap.
