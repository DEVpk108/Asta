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
events on the kernel's `EventBus`. `main.py` registers them in the order
HUD -> Memory -> AI -> Speech -> Voice -> Tools, so the HUD can show `THINKING`
before memory recall and model generation begin.

### Request flow

1. `voice` waits for a wake word (openWakeWord), then captures one utterance
   with streaming Silero VAD.
2. faster-whisper transcribes the audio and the module emits `user_message`.
3. `ai` handles deterministic cases first (tool approvals, conversation mode,
   presentation, screenshot phrasing), then falls back to `IntentRouter`.
4. A command becomes a `tool_request`. The tool layer checks the authority
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

cd hud
npm install
cd ..
```

`requirements-kokoro.txt` is not optional for running the assistant:
`speech/speech_module.py` imports `KokoroEngine` at import time. On an NVIDIA
GPU, install the CUDA build of PyTorch before it (see the note inside that
file).

Optional extras:

```bash
pip install -r requirements-mempalace.txt     # long-term memory
pip install -r voice/requirements-indic.txt   # IndicConformer STT backend
```

## Run

1. Start `llama-server` with your GGUF model. A typical Windows command is:

```powershell
.\\llama-server.exe -m "C:\\Models\\your-model.gguf" --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --jinja --alias asta-local
```

Adjust the model path, context size, and GPU offload for your hardware/model.
`llama-server` provides the OpenAI-compatible `/v1/chat/completions` API used by A.S.T.A.
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

## Configuration

Every setting is read from the process environment. There is no `.env` loader
yet, so export these before starting A.S.T.A.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASTA_DISABLE_HUD` | `0` | Skip the Electron HUD auto-launch |
| `ASTA_PRELAUNCHED_HUD` | `0` | A launcher already started the HUD |
| `ASTA_ENABLE_TEXT_INPUT` | `0` | Enable the legacy terminal text adapter |
| `ASTA_LLM_PROVIDER` | `llama_cpp` | Local LLM provider |
| `ASTA_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server OpenAI-compatible base URL |
| `ASTA_LLM_MODEL` | auto-discovered | llama-server model id/alias |
| `ASTA_LLM_TIMEOUT` | `120` | LLM HTTP timeout in seconds |
| `ASTA_LLM_MAX_OUTPUT_TOKENS` | `256` | Maximum generated tokens per response |
| `ASTA_LLM_REASONING_RETRY_TOKENS` | `512` | Retry budget when the first generation exhausts the output budget |
| `ASTA_HUD_HOST` | `127.0.0.1` | HUD transport bind address |
| `ASTA_HUD_PORT` | `18765` | HUD transport port |
| `ASTA_STT_BACKEND` | `whisper` | `whisper`, `indic` or `hybrid` |
| `ASTA_CHAT_HISTORY_DB` | `data/chat_history.db` | SQLite chat history location |
| `ASTA_MEMPALACE_PATH` | MemPalace default | Long-term memory store path |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | - | Needed for the gated IndicConformer model |

Secrets belong in the environment, never in the repository. `.env`, `*.key` and
`*.pem` are git-ignored.

## Tools and approvals

| Tool | Risk |
| --- | --- |
| `system.close_application`, `vision.screenshot`, `vision.open_screenshot`, `audio.mute`, `audio.unmute` | low |
| `system.open_application`, `system.launch_application` | medium |
| `system.start_process`, `system.stop_process` | high |
| `system.run_command` | critical |

`AuthorityPolicy` runs tools up to **medium** risk automatically. Anything
higher emits `tool_confirmation_required`, A.S.T.A. asks out loud, and the
action only runs after a spoken confirmation. Subprocess calls never use a
shell; they always pass argument lists.

## Layout

| Path | Contents |
| --- | --- |
| `core/` | kernel, event bus, module base, intent router, tool layer |
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

## Status

Early and evolving. The current focus is the voice -> reason -> act -> speak
loop and the HUD. `vision/face_recognition.py` is a standalone OpenCV
experiment and is not wired into the kernel yet.
