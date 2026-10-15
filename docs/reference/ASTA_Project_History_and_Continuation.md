# ASTA Project History and Continuation

Converted reference from `ASTA_Project_History_and_Continuation.docx`. The original document is retained.

A.S.T.A. — Project History & Continuation Document

Recovered project memory after loss of the original development chat

Repository: DEVpk108/Asta  |  Default branch: main  |  Recovery snapshot: 7 September 2026

## 1. Purpose of this document

This document is the durable handoff for the A.S.T.A. project. It exists so development can continue even if the ChatGPT conversation, reasoning history, or planning discussion is lost again. It combines the surviving project context with the current GitHub repository and its recent commit history. It is not a verbatim recreation of the deleted conversation; it is a recovered engineering record and continuation point.

## 2. Project snapshot

| Project name | A.S.T.A. (Asta) — experimental local-first personal AI / AI engineering assistant |
| --- | --- |
| Core ambition | Grow from a voice assistant into a modular personal AI operating system / agent runtime for coding and electronics work |
| Primary language | Python |
| AI runtime | Local LM Studio server; current engine default model: nvidia/nemotron-3-nano-4b |
| Core architecture | Kernel + Event Bus + Modules + Intent Router + Tool system |
| Interaction path | Speech / text input → conversation → intent analysis → routed action or model response |
| Long-term direction | Memory, reflection, proactive suggestions, tool execution, code execution/testing, modular model backends, vision and hardware/electronics integration |
| Repository status | Public GitHub repository, actively evolving |

Current README describes Asta as a lightweight assistant focused on low latency, everyday tasks, notes/reminders, face recognition, OpenCV and a Python kernel/event-bus architecture. The codebase has already moved beyond that description, so the README should be treated as lagging documentation rather than the complete architectural definition.

## 3. Original vision we were working toward

The project was intentionally larger than “make a chatbot.” The recurring design goal was a local-first AI engineering companion that could become a personal AI operating system.

- Local-first execution to minimize dependency on paid cloud APIs and keep the system responsive/private where practical.

- A modular architecture so models, APIs, speech engines and subsystems can be swapped without rebuilding the whole system.

- Long-term memory: storing useful facts, notes, project context and retrieving them when relevant.

- Reflection/review loops: one model or component can generate an answer/code and another component can critique, test or improve it.

- Agentic workflows: A.S.T.A. should be able to reason about a task, call tools, inspect results and continue.

- Code execution/testing feedback: generated code should be testable, with failures/results fed back into the improvement loop.

- Proactive suggestions: A.S.T.A. should eventually surface useful ideas instead of waiting for every instruction.

- Support for both software engineering and electronics work.

- Voice-first interaction through speech recognition, wake word detection and speech output, while retaining direct text interaction.

## 4. Recovered development timeline

| Date | Repository evidence | Meaning / architectural progress |
| --- | --- | --- |
| 29 Aug 2026 | README/kernel architecture work | Python kernel + event-bus structure becomes an explicit architectural direction. |
| 30 Aug 2026 | Implement intent routing in AIModule and add IntentRouter | Intent processing is separated from raw model generation. |
| 31 Aug 2026 | Refactor IntentRouter to return IntentResult objects and improve entity extraction | Intent handling becomes contract-based and structured. |
| 1 Sep 2026 | Architecture diagram for Python kernel/event-bus | Architecture is documented as a system rather than a collection of scripts. |
| 2 Sep 2026 | Two Snake game implementations added | UI/tool experimentation enters the repository; later tool architecture work gives this a cleaner home. |
| 3 Sep 2026 | README and Snake HTML structure refactored | Project documentation and demo/UI work continue in parallel. |
| 5 Sep 2026 | AIModule / IntentRouter refactor; IntentResult, MemoryRecord, ModelRequest and Tool contracts added | The project crosses an important boundary from “working code” to explicit interfaces/contracts between subsystems. |
| 6 Sep 2026 | Tool architecture refactor; AI engine changed to nvidia/nemotron-3-nano-4b | A.S.T.A. gains Tool, ToolDispatcher, ToolRegistry and AuthorityPolicy concepts; model backend remains swappable. |
| 6 Sep 2026 | Ignore .tflite model files | Large/local model artifacts are kept out of source control. |

Important: the exact reasoning and discussion that produced these changes lived partly in the now-deleted chat. The timeline above is therefore a reconstruction from Git history plus surviving project context, not a literal transcript.

## 5. Current architecture (recovered)

The repository now clearly separates a core runtime from higher-level modules and contracts.

A.S.T.A.

│

├── Kernel

│   ├── EventBus

│   ├── IntentRouter

│   └── Module lifecycle

│

├── AI Module

│   ├── Intent analysis

│   ├── Command routing

│   ├── Memory request routing

│   └── Conversational generation

│

├── Contracts

│   ├── Intent / IntentResult

│   ├── MemoryRecord

│   ├── ModelRequest

│   └── Tool

│

├── Tools

│   ├── Tool Registry

│   ├── Tool Dispatcher

│   ├── Authority Policy

│   └── Concrete tools (e.g. EchoTool / future game & system tools)

│

└── Interaction layer

├── Speech / Wake word

├── Conversation

└── Voice / HUD / other modules

The Kernel currently owns the EventBus and IntentRouter, registers modules, initializes them in order, runs a simple loop, and shuts them down in reverse order. AIModule subscribes to user_message and routes COMMAND and MEMORY intents through events instead of sending every request straight to the LLM.

## 6. Contract layer and why it matters

- IntentResult provides structured intent classification output rather than passing opaque strings between components.

- MemoryRecord establishes a stable representation for future long-term memory implementations.

- ModelRequest creates a boundary between application logic and whichever model provider/runtime is used.

- Tool establishes a standard interface so tools can be registered and dispatched without hard-coding every capability into the AI module.

- This contract-first direction is the right foundation for the long-term “swappable modules / models / APIs” goal.

## 7. Tool system direction

The newest architecture introduces an important safety and control boundary: the model should propose or request actions; a separate tool layer decides what can actually execute.

- Tool: defines the capability contract.

- ToolRegistry: stores/discovers available tools.

- ToolDispatcher: receives tool requests and executes the appropriate tool.

- AuthorityPolicy: evaluates authorization/risk before execution.

- EchoTool: currently used as a minimal/reference implementation while the architecture is being established.

This is strategically important because it gives A.S.T.A. a path toward real agentic behavior without turning the LLM into an unrestricted process executor.

## 8. Current AI / LM Studio path

The current AIEngine talks to a local LM Studio server at http://localhost:1234 using the /api/v1/chat endpoint. It streams responses, records time-to-first-token (TTFT), maintains a previous_response_id for conversation continuity, and sends a system prompt for a concise voice-assistant style. The default model in the checked-in engine is nvidia/nemotron-3-nano-4b.

The AI module first asks IntentRouter to analyze the user message. COMMAND produces command_request, MEMORY produces memory_request, and other intents fall back to conversational generation. Sentence-level streaming can emit assistant_sentence events while the final assistant_response is emitted after generation completes.

## 9. Existing / planned interaction capabilities

- Whisper-based speech-to-text is part of the intended voice pipeline.

- Wake-word detection work is present in ai/wakeword, including a dataset structure and openWakeWord resources.

- The README describes computer vision / face detection using OpenCV and lightweight deep-learning approaches such as MobileNet-SSD; verify the live implementation before treating this as fully integrated.

- Speech output has previously used AWS Polly as a temporary implementation, with XTTS discussed as a future local voice option.

- A HUD/voice/module structure exists conceptually; the project should continue moving these behind clean module interfaces.

## 10. What is complete vs. unfinished

### Solid / demonstrably present

- Kernel lifecycle exists.

- Event bus exists.

- IntentRouter exists and is integrated with AIModule.

- Structured contract layer exists.

- Tool system skeleton exists.

- Local LM Studio model backend is active in code.

- Git history is actively capturing architectural changes.

### Partially implemented / needs verification

- Long-term memory architecture is represented by contracts, but a full memory subsystem is not yet established in the recovered snapshot.

- AuthorityPolicy exists as architecture, but policy sophistication and enforcement coverage need a thorough audit.

- Agentic tool calling is not yet a full autonomous loop; the architecture is prepared for it.

- Reflection/review loops are a design goal rather than a clearly completed subsystem.

- Code execution/test feedback loop is a future capability, not yet a mature pipeline.

- Vision, speech output, wake-word and HUD components need integration verification.

## 11. Recommended continuation roadmap

| Stage | What to do |
| --- | --- |
| A — Freeze the architecture | Document contracts and event names. Avoid adding features directly into AIModule that belong in core/services. |
| B — Finish the tool runtime | Define tool request/result schemas, error handling, timeouts, permission/risk levels and deterministic dispatch behavior. |
| C — Build memory subsystem | Implement short-term conversation state and persistent long-term memory behind the MemoryRecord contract. Add retrieval and relevance rules before adding complex embeddings. |
| D — Add controlled agent loop | LLM → plan/request tool → AuthorityPolicy → ToolDispatcher → result → LLM. Add step limits and cancellation. |
| E — Add reflection/testing loop | For coding tasks: generate → execute in sandbox → inspect failure/output → revise → stop when success criteria are met. |
| F — Improve observability | Record latency, TTFT, tool latency, failures, event flow and model/backend identifiers. This is especially useful for the local-first performance goal. |
| G — Clean documentation | Update README so it describes the actual kernel/contracts/tool architecture instead of only the earlier assistant feature set. |
| H — Version the project deliberately | Create clear milestones such as V0.1, V0.2, etc., and keep an architecture changelog in the repo. |

## 12. Exact continuation checkpoint

When returning to A.S.T.A. after a lost chat, start here rather than re-inventing the project.

- Open DEVpk108/Asta on main and inspect core/, especially core/kernel.py, core/event_bus.py, core/intent_router.py, core/contracts/, and core/tools/.

- Preserve the Kernel/EventBus/IntentRouter separation.

- Preserve the contract layer; do not collapse contracts back into implementation-specific classes.

- Preserve ToolRegistry + ToolDispatcher + AuthorityPolicy as the control boundary for tool execution.

- Use the current local LM Studio backend only as a backend implementation; keep the higher-level architecture provider/model agnostic.

- Prioritize finishing the controlled tool loop and memory subsystem before piling on cosmetic features.

- Keep performance measurable: TTFT, end-to-end latency, tool latency and resource use matter to the local-first design.

## 13. How to use this document after future chat loss

Paste or upload this document into a new conversation and say: “This is the A.S.T.A. continuation document. Inspect the current GitHub repo and continue from the checkpoint.” The assistant should then treat this file as project history and GitHub as the implementation source of truth.

## 14. Evidence / source notes

Recovered from: GitHub repository DEVpk108/Asta, current main branch, README, current core directory structure, current AI module / engine files, contract directory, and recent commit messages visible in GitHub on 7 September 2026. Some older architectural intentions come from surviving project context and are explicitly described as recovered context rather than exact historical transcript.
