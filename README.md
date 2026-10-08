# A.S.T.A.

### *Adaptive System for Technical Assistance*

A.S.T.A. is a local-first, voice-first personal AI assistant. It waits for a
wake word, transcribes speech locally, reasons with a local LLM served by llama.cpp, speaks the answer back with a local TTS model, and executes authorized
system actions through a tool layer with risk-based approvals. An Electron HUD
shows what it is doing in real time.

The default path runs on the machine: local wake word, local STT, local LLM,
local TTS.

---

## Architecture

```
                        A.S.T.A.
                            |
                  Kernel  +  EventBus (pub/sub)
                            |
   +--------+--------+------+------+--------+--------+
   |        |        |             |        |        |
  HUD    Memory      AI         Speech    Voice    Tools
```

Modules never call each other directly. They publish and subscribe to named
events on the kernel's `EventBus`. `main.py` registers the runtime modules so
HUD state and task progress can update independently from AI reasoning, voice
capture, speech output, and tool execution.

### Request flow

1. `voice` waits for a wake word (openWakeWord), then captures one utterance
   with streaming Silero VAD.
2. faster-whisper transcribes the audio and the module emits `user_message`.
3. `ai` handles deterministic cases first (tool approvals, conversation mode,
   presentation, screenshot phrasing), then optionally runs the System 1 decision
   provider (Laya) for structured routing signals. The existing `IntentRouter`
   remains authoritative for executable command parsing; opt-in project tasks
   use the cognitive planner and the existing task runtime.
4. A command or planned task step becomes a `tool_request`. The tool layer checks the authority
   policy and either runs it or emits `tool_confirmation_required` and waits
   for a spoken yes/no.
5. Anything else goes to the LLM, which streams sentences back as
   `assistant_sentence` events.
6. `speech` synthesizes each sentence with Kokoro. `hud` mirrors state, chat and
   a live audio envelope to Electron over a localhost JSON-lines socket.

## Requirements

- **Python 3.10+** - the code uses `X | None` annotations that are evaluated at
  runtime.
- **Node.js 18+** for the Electron 31 HUD.
- **llama.cpp `llama-server`**, exposing the OpenAI-compatible API on
  `http://127.0.0.1:8080/v1`. A.S.T.A. discovers the loaded model from
  `/v1/models` unless `ASTA_LLM_MODEL` is set explicitly.
- A microphone and speakers. A CUDA GPU is optional; Whisper and Kokoro fall
  back to CPU.
- Wake-word models in `ai/wakeword/generated/models/`: `hello_asta.onnx`,
  `hey_asta.onnx`, `wake_up_asta.onnx`. `*.onnx` is git-ignored, so these are
  not committed, and `VoiceModule` raises at startup when they are missing.

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-kokoro.txt
# Optional System 1 decision layer
pip install -r requirements-laya.txt

cd hud
npm install
cd ..
```

Then download the low-latency speech models (about 500 MB, stored in the
git-ignored `models/speech/`):

```bash
python scripts/download_speech_models.py
```

This fetches NVIDIA Nemotron 3.5 ASR Streaming (English + Hindi, run on CPU
through sherpa-onnx) and the Smart Turn end-of-turn model. Without them
A.S.T.A. falls back to Whisper and plain silence detection.

`requirements-kokoro.txt` is not optional for running the assistant:
`speech/speech_module.py` imports `KokoroEngine` at import time. On an NVIDIA
GPU, install the CUDA build of PyTorch before it (see the note inside that
file).

Optional extras:

```bash
pip install -r requirements-laya.txt          # Laya System 1 decision layer
pip install -r requirements-mempalace.txt     # long-term memory
pip install -r voice/requirements-indic.txt   # IndicConformer STT backend
pip install -r requirements-browser.txt       # browser automation for capability setup
```

For autonomous browser setup on Windows, the Spotify setup operator uses a
dedicated persistent browser profile. If the optional browser package is
installed, A.S.T.A. can open the Spotify Developer Dashboard, wait for the
user-only login boundary when necessary, create/configure the A.S.T.A. app,
capture its Client ID, persist it to the local ignored `.env`, refresh the
Spotify provider, and resume the original task. Browser automation never
handles the user's Spotify password, MFA codes, or CAPTCHA.

## Run

1. Configure A.S.T.A. to manage the local llama.cpp server. On startup, A.S.T.A.
checks whether the configured loopback server is already running. If it is not,
A.S.T.A. starts `llama-server`, waits for `/v1/models` to become ready, warms
the model, and owns that process for the lifetime of the assistant.

Set the server executable and GGUF model path when they are not automatically discoverable:

```powershell
$env:ASTA_LLAMA_SERVER_PATH="C:\\Models\\llama-server.exe"
$env:ASTA_LLM_MODEL_PATH="C:\\Models\\your-model.gguf"
```

A.S.T.A. first checks the system PATH and the repository's sibling `llama\\`
directory for `llama-server.exe`. If the selected server directory contains
exactly one `models\\*.gguf`, that model is discovered automatically.

The default startup settings use context `8192`, `-ngl 99`, `--jinja`, and
`--reasoning off`. These can be changed through the configuration variables
below.

You can still start llama.cpp yourself. When A.S.T.A. detects an already-running
server, it uses it but does not take ownership of or terminate that external
process.

### Managed LFM2.5-VL vision runtime

A.S.T.A. manages LFM2.5-VL as a separate local llama-server runtime on
`http://127.0.0.1:8090/v1`. The default is **lazy loading**: startup does not
load the vision model or its multimodal projector. The first `vision.inspect`
request starts the server and waits for `/v1/models` before sending the image.

Set `ASTA_VISION_PRELOAD=1` when startup loading is preferred. As with the main
llama.cpp runtime, A.S.T.A. stops the vision server only when it started that
process itself; an already-running local server is reused.

The managed vision runtime expects the LFM2.5-VL GGUF plus its `mmproj` GGUF.
When several GGUFs are present in `ASTA_VISION_MODEL_DIR`, set
`ASTA_VISION_MODEL_PATH` and `ASTA_VISION_MMPROJ_PATH` explicitly.

2. From the repository root:

```bash
python main.py
```

`main.py` launches the Electron HUD first (`npm start` inside `hud/`) and only
then imports the modules that load heavy local models, so the HUD is already
visible while Whisper and Kokoro warm up. Say **"hey asta"**, **"hello asta"**
or **"wake up asta"** to start talking. Ask it to turn conversation mode on to
skip the wake word, and say **"stop"** while it is speaking to interrupt.

The HUD can also be run on its own:

```bash
cd hud
npm start
```

## Incremental voice execution (V1)

A.S.T.A. can optionally begin executing safe actions before the user finishes a
spoken sentence. This is the first step toward the mid-sentence interaction
style demonstrated in the Jev workflow.

When enabled, the voice path becomes:

```text
microphone
  -> Silero VAD
  -> rolling partial Whisper decode
  -> incremental command detector
  -> early action commitment
  -> background dispatch
  -> normal AI / Planner / ToolRuntime
  -> continue listening
  -> final Whisper decode
  -> execute only the uncommitted remainder
```

V1 commits only application-open actions (`open`, `launch`, `start`) after the
same semantic target is seen in consecutive partial transcripts and the target
can be resolved by the generic `ApplicationManager`. It intentionally does not
early-commit destructive actions.

This first implementation uses repeated rolling-window faster-whisper decodes
rather than a native streaming decoder. The goal is to validate the interaction
model without replacing the existing local STT backend. The action detector and
speech session are isolated so a later streaming STT backend can replace the
partial decoder without changing TaskRuntime or the computer-control tools.

Enable it for a local experiment:

```powershell
$env:ASTA_INCREMENTAL_VOICE="1"
python main.py
```

Useful tuning variables:

```text
ASTA_INCREMENTAL_STABLE_UPDATES   # default 2
ASTA_INCREMENTAL_STT_INTERVAL_MS  # default 800
ASTA_INCREMENTAL_MIN_AUDIO_MS     # default 850
ASTA_INCREMENTAL_STT_WINDOW_MS    # default 5000
```

A representative test utterance is:

```text
"Hey Asta, open up Chrome for me and once you're there search for Christopher Nolan"
```

A successful incremental trace should commit the application-open prefix while
the user is still speaking, then process the remaining search instruction only
after the final utterance decode.

## Configuration

Every setting is read from the process environment. On startup, `main.py`
also loads a `.env` file from the repository root (see `.env.example`); values
already set in the environment take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASTA_DISABLE_HUD` | `0` | Skip the Electron HUD auto-launch |
| `ASTA_PRELAUNCHED_HUD` | `0` | A launcher already started the HUD |
| `ASTA_ENABLE_TEXT_INPUT` | `0` | Enable the legacy terminal text adapter |
| `ASTA_AGENT_MODE` | `0` | Enable the cognitive planner for scoped, open-ended project tasks |
| `ASTA_WORKSPACE_PATH` | Git root, otherwise current directory | Active project root for workspace file tools and command working directories |
| `ASTA_LLM_PROVIDER` | `llama_cpp` | Local LLM provider |
| `ASTA_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server OpenAI-compatible base URL |
| `ASTA_LLM_MODEL` | auto-discovered | llama-server model id/alias |
| `ASTA_LLM_MODEL_PATH` | - | GGUF file path used when A.S.T.A. auto-starts llama-server |
| `ASTA_LLAMA_SERVER_PATH` | `llama-server` on PATH | llama-server executable used for auto-start |
| `ASTA_LLM_AUTOSTART` | `1` | Automatically start a local llama-server when it is not already running |
| `ASTA_LLAMA_SERVER_STARTUP_TIMEOUT` | `120` | Seconds to wait for the managed llama-server to become ready |
| `ASTA_LLM_CONTEXT_SIZE` | `8192` | llama.cpp context size when A.S.T.A. starts the server |
| `ASTA_LLM_GPU_LAYERS` | `99` | llama.cpp GPU layer offload when A.S.T.A. starts the server |
| `ASTA_LLM_JINJA` | `1` | Pass `--jinja` to llama-server |
| `ASTA_LLM_REASONING` | `off` | Pass the reasoning mode to llama-server |
| `ASTA_VISION_BASE_URL` | `http://127.0.0.1:8090/v1` | LFM2.5-VL OpenAI-compatible endpoint |
| `ASTA_VISION_MODEL` | auto-discovered | Vision server model id/alias |
| `ASTA_VISION_MODEL_PATH` | auto-discovered | LFM2.5-VL GGUF path |
| `ASTA_VISION_MMPROJ_PATH` | auto-discovered | LFM2.5-VL multimodal projector GGUF path |
| `ASTA_VISION_MODEL_DIR` | `E:\\Projects\\llama\\models` | Shared local vision model directory |
| `ASTA_VISION_SERVER_PATH` | `E:\\Projects\\llama\\llama-server.exe` | llama-server executable for managed vision runtime |
| `ASTA_VISION_PRELOAD` | `0` | Load LFM2.5-VL at startup (`1`) or lazily on first vision request (`0`) |
| `ASTA_VISION_CONTEXT_SIZE` | `8192` | Context size for the managed vision server |
| `ASTA_VISION_GPU_LAYERS` | `99` | GPU layer offload for the managed vision server |
| `ASTA_VISION_MMPROJ_OFFLOAD` | `1` | GPU-offload the multimodal projector |
| `ASTA_LLM_TIMEOUT` | `120` | LLM HTTP timeout in seconds |
| `ASTA_LLM_MAX_OUTPUT_TOKENS` | `256` | Maximum generated tokens per response |
| `ASTA_LLM_REASONING_RETRY_TOKENS` | `512` | Retry budget when the first generation exhausts the output budget |
| `ASTA_DECISION_ENGINE` | `disabled` | System 1 decision provider: `disabled` or `laya` |
| `ASTA_LAYA_MODEL` | `multilingual` | Laya checkpoint: `multilingual` or `english` |
| `ASTA_LAYA_DEVICE` | `auto` | Laya device override such as `cpu` or `cuda` |
| `ASTA_LAYA_PRELOAD` | `1` | Preload the selected Laya checkpoint during A.S.T.A. startup |
| `ASTA_LAYA_MAX_LOADED` | `1` | Maximum Laya checkpoints kept resident by its router |
| `ASTA_HUD_HOST` | `127.0.0.1` | HUD transport bind address |
| `ASTA_HUD_PORT` | `18765` | HUD transport port |
| `ASTA_HUD_TOKEN` | random per run | Shared secret the HUD must present; generated automatically by `main.py` / the launcher |
| `ASTA_LLM_HISTORY_TURNS` | `12` | Conversation turns replayed to the LLM |
| `ASTA_LLM_HISTORY_CHARS` | `12000` | Character budget for replayed conversation history |
| `ASTA_APPROVAL_TTL_SECONDS` | `30` | Seconds a pending tool approval stays valid |
| `ASTA_INCREMENTAL_STT_BEAM_SIZE` | `1` | Whisper beam size for incremental partial decodes |
| `ASTA_STT_BACKEND` | `auto` | `auto` (Nemotron when downloaded, else Whisper), `nemotron`, `whisper`, `indic` or `hybrid` |
| `ASTA_STT_LANGUAGE` | `en+hi` | Nemotron language. `en+hi` decodes English and Hindi in parallel and keeps the transcript that reads as real English or real Hindi (Nemotron's own `auto` writes Indian-accented English in Devanagari). Also `auto`, `en`, `hi`, ... For Whisper, `auto` enables language detection |
| `ASTA_NEMOTRON_MODEL_DIR` | `models/speech/...-560ms-...` | Nemotron model folder (pick another chunk size with `download_speech_models.py --chunk`) |
| `ASTA_NEMOTRON_THREADS` | `4` | CPU threads for Nemotron |
| `ASTA_NEMOTRON_PROVIDER` | `cpu` | `cpu`, or `cuda` with the GPU build of sherpa-onnx |
| `ASTA_SMART_TURN` | `1` | Smart Turn end-of-turn detection when its model is downloaded |
| `ASTA_SMART_TURN_SILENCE_MS` | `200` | Silence before Smart Turn checks whether you finished |
| `ASTA_SMART_TURN_THRESHOLD` | `0.5` | Completion probability needed to end the turn early |
| `ASTA_VAD_SILENCE_MS` | `700` | Silence that always ends a turn (used when Smart Turn says "not finished" or is off) |
| `ASTA_TTS_VOICE` | `am_michael` | Kokoro English voice |
| `ASTA_TTS_HINDI` | `1` | Speak Devanagari replies with Kokoro's Hindi pipeline |
| `ASTA_TTS_HINDI_VOICE` | `hf_alpha` | Kokoro Hindi voice (`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`) |
| `ASTA_TTS_HINDI_PRELOAD` | `1` | Load the Hindi voice in the background at startup |
| `ASTA_APPLE_MUSIC_APP` | `Apple Music` | App name ASTA opens for "play … on Apple Music" |
| `ASTA_APPLE_MUSIC_PROCESS` | `AppleMusic` | Apple Music process name (UI Automation search + play); inspect with `python scripts/uia_dump.py AppleMusic` |
| `ASTA_INTENT_LLM` | `1` | Let the local LLM turn free phrasing the rule router misses ("Baithi Hai is up now, put it on") into one structured action, grounded in the most recent task; `0` = rules only |
| `ASTA_INTENT_LLM_TIMEOUT` | `8` | Seconds to wait for that intent pass before falling back to chat |
| `ASTA_TTS_FIRST_CLAUSE` | `1` | Speak a reply's first clause on its own so audio starts sooner |
| `ASTA_TTS_OUTPUT_LATENCY` | `low` | Audio output buffer (`low`, `high` or seconds) |
| `ASTA_CHAT_HISTORY_DB` | `data/chat_history.db` | SQLite chat history location |
| `ASTA_VOICE_POST_TTS_GUARD_MS` | `80` | Short post-TTS settle window; speech during it is retained as VAD preroll |
| `ASTA_INCREMENTAL_VOICE` | `0` | Enable mid-sentence safe action commitment |
| `ASTA_INCREMENTAL_STABLE_UPDATES` | `2` | Consecutive matching partial STT updates required before early commit |
| `ASTA_INCREMENTAL_STT_INTERVAL_MS` | `800` | Partial STT polling interval in milliseconds |
| `ASTA_INCREMENTAL_MIN_AUDIO_MS` | `850` | Minimum captured speech before partial STT begins |
| `ASTA_INCREMENTAL_STT_WINDOW_MS` | `5000` | Rolling audio window used for partial STT |
| `ASTA_VAD_PRE_ROLL_MS` | `900` | Audio retained before VAD onset so first spoken words are not clipped |
| `ASTA_MEDIA_DEFAULT_PROVIDER` | - | Optional default provider for media queries without an explicit provider |
| `ASTA_SPOTIFY_CLIENT_ID` | - | Spotify developer app client ID for authenticated track playback |
| `ASTA_SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8765/callback` | Loopback URI used by Spotify PKCE authorization |
| `ASTA_SPOTIFY_TOKEN_PATH` | `data/spotify_token.json` | Local Spotify OAuth token cache |
| `ASTA_MEMPALACE_PATH` | MemPalace default | Long-term memory store path |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | - | Needed for the gated IndicConformer model |

Secrets belong in the environment, never in the repository. `.env`, `*.key` and
`*.pem` are git-ignored.

## Laya System 1 decision layer

A.S.T.A. can optionally run Laya as a fast, non-generative decision layer before
the existing intent/parser and reasoning paths.

Enable it with:

```powershell
$env:ASTA_DECISION_ENGINE="laya"
pip install -r requirements-laya.txt
```

A.S.T.A. uses Laya's **multilingual checkpoint by default**. The selected model is
passed explicitly to Laya rather than letting the router switch checkpoints per
request. This keeps the System 1 path predictable and avoids loading the larger
English checkpoint when it is not needed.

When Laya is enabled, its selected checkpoint is loaded during A.S.T.A. startup,
while the HUD is already visible. On shutdown, A.S.T.A. unloads the checkpoint.

The model can be overridden:

```powershell
$env:ASTA_LAYA_MODEL="english"
```

Set `ASTA_LAYA_MODEL=multilingual` to return to the default.

When `ASTA_LAYA_PRELOAD=1`, A.S.T.A. preloads only the selected checkpoint so
the first user request does not pay the cold model-load cost. `ASTA_LAYA_MAX_LOADED`
still controls the router's resident-model limit.

The first integration is deliberately advisory: Laya emits a `decision_result`
event containing structured decisions such as intent, domain, tool need,
reasoning need, sensitivity, selected checkpoint, and latency. The existing
rule-based `IntentRouter`, Planner, ToolRuntime, and AuthorityManager remain
authoritative. This lets A.S.T.A. benchmark Laya against the current pipeline
before promoting any Laya decision to control execution.

The same lifecycle principle applies to llama.cpp: A.S.T.A. owns a server only
when it started that process itself. A server started externally is reused but
never terminated by A.S.T.A.

Laya is not a replacement for the main local LLM. It is intended as a System 1
routing/classification layer that can later feed model selection, planning
strategy, multilingual intent classification, and guardrails.

## Media control

A.S.T.A. exposes media playback through a provider abstraction rather than
hardcoding a single application.

- `media.control` handles play, pause, toggle, next, previous and stop.
- Windows transport controls use the global media keys when a provider API is
  not required.
- Spotify track playback uses the Spotify Web API when `ASTA_SPOTIFY_CLIENT_ID`
  is configured. A.S.T.A. uses Authorization Code with PKCE and stores the
  refresh token under `data/spotify_token.json`.

For Spotify track playback, create a Spotify developer app and allowlist the
loopback redirect URI `http://127.0.0.1:8765/callback`, then set:

```powershell
$env:ASTA_SPOTIFY_CLIENT_ID="your_client_id"
$env:ASTA_SPOTIFY_REDIRECT_URI="http://127.0.0.1:8765/callback"
```

The first Spotify play request opens the browser for authorization. After the
one-time authorization, a command such as **Play Hanuman Chalisa on Spotify**
can search the Spotify catalog, select a matching track deterministically, and
start playback on the active Spotify device.

Spotify currently requires Premium for playback-control APIs, and Spotify's
Development Mode also requires the app owner to have Premium.

## Tools and approvals

| Tool | Risk |
| --- | --- |
| `system.close_application`, `vision.screenshot`, `vision.open_screenshot`, `audio.mute`, `audio.unmute` | low |
| `system.open_application`, `system.launch_application`, `computer.click` | medium |
| `system.start_process`, `system.stop_process`, `computer.type_text`, `computer.keypress`, `computer.hotkey` | high |
| `system.run_command` | critical |
| `filesystem.list_files`, `filesystem.read_file` | low |
| `filesystem.write_file` | medium |

`AuthorityPolicy` runs tools up to **medium** risk automatically. Anything
higher emits `tool_confirmation_required`, A.S.T.A. asks out loud, and the
action only runs after a spoken confirmation. Subprocess calls never use a
shell; they always pass argument lists.

Approvals are deliberately strict:

- A pending approval expires after `ASTA_APPROVAL_TTL_SECONDS` (default 30s).
- Only explicit answers such as "yes", "confirm" or "go ahead" approve; bare
  fillers like "ok" or "sure" do not.
- Saying anything else cancels the pending request, so an unrelated "yes"
  later can never run it.

Keyboard tools are `high` risk because a key sequence such as Win+R, typing a
command and Enter amounts to arbitrary command execution. To trust them for
your own setup, grant them explicitly through `AuthorityManager.grant(...)`.

### Workspace-scoped project tasks (early preview)

Project requests such as creating or testing a code file can use the existing
task planner and runtime when `ASTA_AGENT_MODE=1`. Set
`ASTA_WORKSPACE_PATH` to the project directory you want A.S.T.A. to work in;
otherwise the workspace runtime attempts to use the current Git repository.
The workspace path is an explicit boundary for file tools and the working
directory for `system.run_command`.

The initial project-file surface is intentionally small:

- `filesystem.list_files` lists workspace entries while omitting common
  credential and dependency directories.
- `filesystem.read_file` reads UTF-8 text files up to 512 KiB.
- `filesystem.write_file` creates or atomically updates a UTF-8 file up to
  512 KiB. Existing files must first be read and updated with that read's
  SHA-256 digest, so stale content is not silently overwritten.
- `system.run_command` accepts an executable and argument list (no shell),
  uses a workspace-relative `cwd`, and applies the tool request timeout.
  It remains **critical risk** and requires confirmation under the default
  policy.

The project planner requires a verified read-back after a file write and
checks the command's observed return code and stdout when a project plan runs
a command. If that evidence is missing, A.S.T.A. must not report the task as
verified. Tests use deterministic plans and temporary workspaces:

```bash
pytest -q tests/test_workspace_project_task.py
```

**Important:** Workspace-rooted paths and working directories are not an
operating-system sandbox. An approved executable can still access other local
files or the network. Do not treat `system.run_command` as safe for untrusted
code; a true restricted runner/container remains future work. The first file
tools also do not delete, move, or patch files.

### HUD transport security

The HUD talks to the runtime over `127.0.0.1:18765`. Every client must first
send `{"type": "hud.hello", "token": "<ASTA_HUD_TOKEN>"}`; unauthenticated
clients, and anything that is not a JSON line (for example an HTTP request sent
by a web page), are disconnected before they can send commands or read state.

## Layout

| Path | Contents |
| --- | --- |
| `core/` | kernel, event bus, module base, intent router, decision providers, tool layer |
| `ai/` | llama.cpp client/provider, AI module, runtime patches, wake-word assets |
| `voice/` | microphone, wake word, VAD and STT engines |
| `speech/` | Kokoro TTS and the speech worker |
| `hud/` | Electron HUD and its localhost transport |
| `memory/` | optional MemPalace long-term memory |
| `chat_history/` | local SQLite conversation history |
| `vision/` | screenshot backend, experimental face recognition |
| `input/` | optional terminal text adapter |
| `tests/` | pytest suite |

## Tests

```bash
pytest
```

## Project direction

A.S.T.A. is being built toward a local-first personal AI system that can take a
goal, work out the steps, use the capabilities available to it, check what
happened, and report back. The intended loop is:

```text
UNDERSTAND -> PLAN -> ACT -> OBSERVE -> EVALUATE -> ADJUST / CONTINUE -> REPORT
```

The LLM is the reasoning layer; registered capabilities do the work; the
runtime owns execution state and permissions; observation checks outcomes; and
the HUD presents status without becoming the core orchestration layer. This is
the long-term direction, not a claim that every capability below is already
complete.

## Development roadmap

This roadmap reflects the project direction in [Future Plan & Vision](docs/ASTA_Future_Plan_and_Vision.docx)
and the [Step-by-Step Build Roadmap](docs/ASTA_Step_by_Step_Build_Roadmap.docx).
The repository already contains parts of the tool, task-runtime, planning,
workspace, memory, and approval architecture. The next goal is to make those
parts work together reliably—not to rebuild them or add features for their own
sake.

| Stage | Focus | Completion signal |
| --- | --- | --- |
| Stabilize the foundation | Preserve the voice, AI, HUD, conversation-history, and tool flow; keep startup and shutdown predictable; document event contracts and add useful runtime traces. | A repeatable baseline works across a normal session, shutdown, and restart. |
| Prove the general agent loop — next major milestone | Use generic, permission-aware tools to inspect a small project, create or modify a file, run a controlled command or test, inspect the result, and recover from a bounded, safe failure. | A.S.T.A. completes a small multi-step project task without a special hard-coded script for that exact request, then reports what it changed and what it verified. |
| Build project/workspace continuity | Track project identity, root, repository and branch, current task, important files, and recent changes so a request can resume in the right environment. | A project follow-up is grounded in the registered workspace and its current state. |
| Grow memory in deliberate layers | Keep chat history separate from useful episodic, project, semantic, user, and procedural memory. Preserve source and confidence where useful; support summarizing and forgetting instead of treating every old statement as permanently true. | A.S.T.A. retrieves relevant project knowledge without indiscriminately storing every conversation. |
| Improve planning, reflection, and proactivity | Decompose larger goals, checkpoint progress, observe results, replan within explicit limits, and save only useful lessons. Make proactive suggestions only when supported by relevant context. | Failures are diagnosed and handled safely; the assistant knows when to stop or ask the user. |
| Expand integrations and specialist capabilities | Add domain-focused agents, external APIs/services, broader desktop workflows, and eventually electronics or physical devices after the core loop is dependable. | New capabilities plug into stable contracts without rewriting the core or granting excessive authority. |

**Safety applies at every stage.** New capabilities should declare their inputs,
outputs, risk, permissions, and execution boundaries. Keep approvals,
timeouts, resource limits, stop conditions, and an understandable action trail
in the runtime as capabilities expand.

## Development principles

- Build one useful increment at a time; keep each version working.
- Define the capability contract before writing a large implementation.
- Test happy paths, failures, edge cases, and the real voice/HUD workflow.
- Prefer modular, observable, reversible changes and local-first operation.
- Add capabilities through tools instead of growing a large set of special-case
  conversation branches.
- Prove the single-agent tool loop and permission boundaries before investing
  in a large multi-agent framework, complex long-term retrieval, or unrestricted
autonomy.

## Current status

Early and evolving. The current focus remains the voice -> reason -> act ->
speak loop and the HUD. The next practical proof point is a dependable
small-project workflow: inspect a project, make a requested change, run it,
read the result, recover safely if it fails, and report the outcome. The roadmap
is staged; it does not imply that every planned capability is implemented or
reliable today. `vision/face_recognition.py` remains a standalone OpenCV
experiment and is not wired into the kernel.
