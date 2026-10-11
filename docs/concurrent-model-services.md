# Concurrent conversation and background project work

## Scope

Optional two-profile local inference: LFM2.5-2.6B for interactive conversation
and Qwen3.5-4B for project planning and bounded agent decisions. Model names are
configuration, not hard-coded architecture. The router is disabled by default.
This patch is standalone against main at 6afda1b; do not apply the earlier
ASTA-on-demand-model-router.patch first.

Two services do not provide two GPUs. Dual mode allows overlapping HTTP
inference on separate local llama.cpp servers. It does not guarantee overlapping
CUDA kernels, a particular response latency, or enough VRAM with speech/vision.
The coding service starts lazily and stays resident after its first request.

## Operating modes

- `ASTA_MODEL_ROUTER=0`: unchanged legacy single-model provider.
- `ASTA_MODEL_ROUTER=1`, `ASTA_MODEL_MODE=dual`: independent general/coding
  service locks, separate client histories and request sessions; one request per
  profile at a time. Interactive chat/intent JSON always use general, even when
  a coding task is active. New coding inference waits for existing foreground
  requests at inference boundaries; active generations are not preempted.
- `ASTA_MODEL_MODE=on_demand`: explicit low-memory fallback, one text service at
  a time. Chat can wait behind coding inference in this mode. No automatic
  unload/retry policy after an OOM: errors remain visible and the user chooses
  the fallback.
- `ASTA_BACKGROUND_TASKS=1`: planning and the existing synchronous project
  tool loop run on a dedicated worker, including approved continuations. One
  project task only. No new arbitrary shell execution or parallel tool workers.

## Ownership, safety and limitations

Both GGUF paths must be explicit. Endpoints must be distinct loopback HTTP
ports. A mismatched model identity is rejected. Only ASTA-owned child processes
are stopped. Dual mode may reuse an externally owned endpoint if its model
identity matches; its lifecycle belongs to the user. On-demand mode refuses to
stop an external model in order to switch profiles.

The existing TaskManager has one active slot. While background project work
owns it, other tool actions and new project tasks are held, not interleaved.
Read-only conversation, task status and approvals remain available. This is not
a multi-project agent or concurrent desktop automation implementation.

Existing permissions and critical sandbox approvals remain mandatory. Ordinary
chat does not cancel a background approval, and bare yes/no cannot authorize it.
Use the HUD button (request-ID bound), `approve coding step`, or `reject coding
step`. Approvals still expire according to the existing TTL; expired requests
cannot execute. Terminal background tasks discard their pending approvals.

`cancel coding task` / `cancel project task` stop future work and remove pending
approvals. An already-running tool, filesystem operation or sandbox process is
not forcibly interrupted or rolled back. Do not assume cancellation undoes edits.
A workspace root change requests cancellation before further background steps.
Planning/task state is in memory only; tasks do not resume after restart.

The HUD shows both model-service states plus a separate project phase. Background
progress does not overwrite ordinary listening/speaking state. Approval continues
to use the existing prominent approval flow. No raw model reasoning is displayed.

## Setup

Close ASTA. Keep the existing CUDA llama-server path in `.env`; use a recent
llama.cpp build supporting both models and their chat templates. Download:

- `LiquidAI/LFM2.5-2.6B-GGUF`, `LFM2.5-2.6B-Q4_K_M.gguf`
- `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf`

Install the optional `huggingface_hub` downloader dependency, then run
`scripts/download_model_services.py --model-dir PATH`. The checked-in manifest
pins each download to a repository revision; no model weights are in this patch.

Run `scripts/configure_model_services.py --general PATH --coding PATH`. It updates
only model-related `.env` keys, removes duplicate selected keys, preserves
unrelated configuration, and enables the cognitive planner and background work.
It does not print secrets. `.env` remains ignored. Start with 4K contexts,
256/128 batch/microbatch and one slot per service. Coding thinking is initially
disabled for bounded JSON control; benchmark reasoning separately before
raising context/output budgets. The research benchmarks are not reproduced by
these conservative settings.

## Acceptance checks

1. Run the focused tests for model router, concurrency, server manager, provider,
   HUD, workspace tasks and sandbox. They use fakes, not downloaded models.
2. With ASTA closed, run `python scripts/smoke_concurrent_models.py`. It checks
   model identities, dual residency and a real chat finishing during coding.
   It executes no generated code and shuts down only its own servers. Successful
   output ends with `ASTA_CONCURRENT_MODELS_SMOKE_OK`. Very short coding output
   may finish too soon to demonstrate overlap; that is reported as inconclusive,
   not a false positive.
3. Start ASTA, request an existing supported project task, then ask an unrelated
   conversational question. Check that chat answers while the project phase
   remains visible. Check `project task status` and cancel behavior.
4. At an approval prompt, ask an unrelated question; verify no tool ran. Bare
   `yes` must not approve background work. Use the request-bound approval button
   or the explicit phrase, then verify continuation runs on the project worker.
5. Test memory with speech active and vision loaded when needed; sample
   `nvidia-smi` peak VRAM and response latency. The model-only smoke omits those
   resource consumers. If dual residency fails, set `ASTA_MODEL_MODE=on_demand`
   and use the switching smoke (`scripts/smoke_model_router.py`) instead.

No local Windows/RTX measurement is claimed by the patch. Automated tests prove
software coordination, not model quality or laptop capacity.
