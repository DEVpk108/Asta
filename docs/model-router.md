> This document describes the **on_demand fallback mode**. For dual-model
> operation and background project work, use [concurrent model services](concurrent-model-services.md).
> Both modes ship in one standalone patch; the earlier router patch is not required.

# On-demand general/coding model router (V1)

## Scope

The router is opt-in (`ASTA_MODEL_ROUTER=1
ASTA_MODEL_MODE=on_demand`). Existing single-model behavior is
unchanged when disabled. It reuses llama.cpp, the provider interface, and the
existing validated planner/tools; it is not a second agent framework.

V1 uses conservative rules over the original user goal, plus the existing
workspace task scope. Coding planning and post-action decisions are covered,
not just ordinary chat. Internal capability descriptions are not classified as
user intent. An additional LLM classifier is intentionally not implemented yet.

One kernel-scoped runtime owns model-server lifecycles and serializes all text
inference, including streaming and stateless intent JSON requests. Chat and
agent clients have separate prompts/histories. Chat history is replayed across
model switches; task state, approvals, tool results, and verification remain in
the existing kernel/runtime. A task's role is retained by its in-memory task ID.
Short coding follow-ups and intent classification can reuse the resident model.
An unrelated general question can switch back. Nothing automatically resumes a
previous task or persists model-routing state across restart.

## Setup

Close ASTA normally before changing settings or running the standalone smoke.
Use an up-to-date CUDA llama.cpp build supporting the existing server flags.
Download the chosen coding GGUF yourself; ASTA never downloads models at runtime.
For the initial 8 GB RTX 4060 laptop target, use:

- General: NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf (existing model).
- Coding baseline: Qwen3.5-4B-Q4_K_M.gguf from
  unsloth/Qwen3.5-4B-GGUF.

Before placing a second GGUF in the model directory, set the general model path
explicitly. Automatic single-GGUF discovery becomes ambiguous with two files.
Add/update the following in the ignored local `.env`, using your actual paths:

```text
ASTA_MODEL_ROUTER=1
ASTA_LLM_MODEL_PATH=E:\Projects\llama\models\NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf
ASTA_CODING_LLM_MODEL_PATH=E:\Projects\llama\models\Qwen3.5-4B-Q4_K_M.gguf
ASTA_CODING_LLM_BASE_URL=http://127.0.0.1:8081/v1
ASTA_CODING_LLM_CONTEXT_SIZE=4096
ASTA_CODING_LLM_GPU_LAYERS=99
ASTA_CODING_LLM_BATCH_SIZE=256
ASTA_CODING_LLM_UBATCH_SIZE=128
ASTA_CODING_LLM_PARALLEL=1
ASTA_CODING_LLM_FLASH_ATTN=on
ASTA_CODING_LLM_REASONING=off
ASTA_CODING_LLM_TEMPERATURE=0.6
ASTA_CODING_LLM_MAX_OUTPUT_TOKENS=2048
ASTA_CODING_LLM_MAX_REPLY_SECONDS=120
ASTA_CODING_AGENT_MAX_OUTPUT_TOKENS=2048
```

The existing `ASTA_LLAMA_SERVER_PATH` executable is shared. The general endpoint
remains port 8080. Coding has its own `ASTA_CODING_LLM_*` settings; it does not
inherit the general model ID, model path, batch sizes, or output budget. An
optional `ASTA_CODING_LLM_MODEL` alias must match the server's reported model ID.
Keep both endpoints on loopback HTTP and separate ports. Context includes both
input and output; these initial settings are for small tasks, not full-repository
prompts or guaranteed model quality.

If a profile's target files are missing or its endpoint is already occupied by
an external coding server, reject the switch before unloading the working model.
ASTA only stops its own child process and confirms exit before starting another.
An existing matching external general endpoint can be reused, but ASTA refuses
to unload it for a coding switch. Stop that independently owned server yourself,
or disable routing. A failed switch never silently uses another model.

Idle optional vision resources use the existing GPU-share release mechanism.
Active vision, speech GPU usage, unrelated CUDA processes, and display resources
are not forcibly stopped. Therefore the one-text-model policy is not a guarantee
that the coding model fits the complete ASTA workload. Measure VRAM and latency
on the actual laptop. No claim of live GPU validation is made by unit tests.

The HUD shows the requested profile, loading/ready/error status, and model file
name only. These fields do not overwrite task/approval status and do not display
hidden reasoning, local model-directory paths, or file contents.

## Validation

```powershell
.venv\Scripts\python -m pytest -q tests/test_model_router.py tests/test_llm_provider.py tests/test_llama_server_manager.py tests/test_ai_engine.py tests/test_cognitive_planner.py tests/test_workspace_project_task.py tests/test_hud_module.py
```

With both GGUFs downloaded and ASTA/independent text servers closed:

```powershell
.venv\Scripts\python scripts/smoke_model_router.py
```

Expected: general -> coding -> general, only one owned text-server endpoint
active after each request, ending with `ASTA_MODEL_ROUTER_SMOKE_OK`. This is a
load/switch smoke, not a coding-correctness test and not the full voice/HUD memory
budget. It does not execute generated code, register tools, or write project files.

Then start `main.py`, ask an ordinary question, ask a coding question, and ask
another ordinary question. Verify loading status in the HUD and use `nvidia-smi`
to record memory with speech/vision present. Finally repeat the small-project
acceptance task with sandbox approval and observed pytest results. A better
model does not bypass execution permission or make an unverified task successful.

Rollback: set `ASTA_MODEL_ROUTER=0`, close/restart ASTA. Keep the explicit general
model path after downloading a second GGUF.
