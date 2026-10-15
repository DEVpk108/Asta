# ASTA Project History Engineering Continuation Record

Converted reference from `ASTA_Project_History_Engineering_Continuation_Record.docx`. The original document is retained.

A.S.T.A. — Project History & Engineering Continuation Record

Recovered project context • GitHub-backed state • Engineering handoff

| Purpose of this document<br>This is the durable engineering memory of A.S.T.A. for future development. It records the project's intended identity, recovered architecture, current implementation state, recent engineering history, roadmap, and the exact checkpoint from which development should continue. |
| --- |

## 1. Project Identity

A.S.T.A. is intended to become a personal agentic AI-based Cognitive OS / assistant: a local-first assistant that can handle day-to-day tasks and also help with software, electronics/ECE, computer-science, and other projects. The long-term mental model is closer to JARVIS than to a conventional chatbot.

The project should be treated as an evolving operating environment for intelligence, not merely a voice UI. The architecture is deliberately modular so models, capabilities, and infrastructure can be replaced or expanded.

## 2. Core Vision

- Local-first operation with local models whenever practical, reducing dependence on paid hosted APIs.

- A modular kernel with swappable AI/model providers and independent capability modules.

- Conversation, memory, planning, tools, vision, speech, telemetry, and UI/HUD as separate concerns.

- Agentic behavior with controlled tool access rather than giving the model unrestricted system authority.

- Ability to assist with both ordinary personal tasks and engineering/project work.

- Long-term evolution toward a genuine AI operating environment.

## 3. Recovered High-Level Architecture

The original conceptual architecture was recorded as:

ASTA OS

└─ Cognitive Kernel

├─ Memory

├─ Planner

└─ Scheduler

└─ Assistant Engine

├─ Speech

├─ Vision

└─ Tools

└─ Event / Message Bus

├─ HUD

├─ Telemetry

└─ Plugins

## 4. Current Implemented Kernel Architecture

The current repository has moved from the original simple assistant framing toward a kernel/event-driven design. The GitHub repository currently contains a Kernel, EventBus, IntentRouter, Module abstraction, contracts, and a tools subsystem.

| Component | Current role | State |
| --- | --- | --- |
| Kernel | Owns lifecycle, module registration, startup/shutdown, and core infrastructure. | Implemented |
| EventBus | Decouples modules through events/messages. | Implemented |
| Module | Common lifecycle abstraction for kernel modules. | Implemented |
| IntentRouter | Analyzes user requests and produces typed intent results. | Implemented / evolving |
| AI Module | Receives user messages, routes command/memory intents, or sends conversational work to the model engine. | Implemented |
| AI Engine | Streams responses from local LM Studio and tracks conversation state. | Implemented / tuning |
| Contracts | Defines shared types for intent, memory, model requests, and tools. | Implemented |
| Tool system | Tool contract + registry + dispatcher + authority policy structure. | Recently added |
| Speech / wake-word stack | Voice interaction path with STT/TTS related components. | Partially implemented |
| HUD | Visual state/interface layer. | Planned / not yet complete |

## 5. Current Voice / AI Flow

- Audio input → VAD / wake-word detection → speech-to-text.

- Recognized user message enters the EventBus as a user-message event.

- A.I. module asks IntentRouter to classify/understand the request.

- Command intent → command_request event.

- Memory intent → memory_request event.

- Conversation/task/unknown intent → AI engine generation.

- AI engine streams response deltas and emits assistant_sentence events.

- Final response is emitted as assistant_response for downstream speech/HUD components.

The current A.I. module explicitly follows this routing pattern in the repository, rather than sending every message directly to the LLM.

## 6. Local LLM / LM Studio Architecture

The current AI engine is configured for LM Studio at localhost and uses the local model "nvidia/nemotron-3-nano-4b". It uses streaming generation, stores conversation state, and keeps a previous response identifier so subsequent turns can continue the conversation.

- Streaming is enabled so latency can be measured and first output can be surfaced early.

- The engine records time-to-first-token (TTFT) using a monotonic performance timer.

- Sentence buffering allows incremental downstream TTS rather than waiting for the complete answer.

- The system prompt asks ASTA to behave as a concise, natural local voice assistant.

| Important recovered debugging context<br>One saved project note identifies an LM Studio endpoint mismatch: the code was using the native /api/v1/chat endpoint, while LM Studio's chat UI uses the OpenAI-compatible endpoint. This was specifically identified as a source of behavior/overhead differences when comparing TTFT between the LM Studio chat UI and the Python project. |
| --- |

## 7. Context Management Strategy

The intended context system is not to stuff the full old conversation into every prompt. Instead, project notes describe storing context in a vector database and retrieving only the relevant information when needed.

- Vector database stores persistent contextual knowledge.

- Retrieval should return only the information needed for the current task.

- Prompts are conceptually ranked by seriousness/complexity.

- Context-memory allocation should change dynamically with prompt rank.

- Low-seriousness prompts such as greetings should use less context.

- Medium/high ranks should allocate more reasoning context for coding and difficult tasks.

## 8. Tool / Authority Architecture

A major recent architectural step was separating tool capability from orchestration and authorization. The repository history shows the introduction of Tool, ToolDispatcher, ToolRegistry, AuthorityPolicy, and a sample EchoTool.

- Tool: a capability with a defined contract.

- ToolRegistry: manages available tools.

- ToolDispatcher: executes tool requests and applies authorization checks.

- AuthorityPolicy: centralizes action/risk authorization logic.

- Goal: allow agentic behavior while keeping system actions bounded and auditable.

## 9. Recovered Component Ownership Rules

| Component | Owns | Does NOT own |
| --- | --- | --- |
| Intent Engine | Understanding request | Model selection, execution |
| Memory | Storing/retrieving knowledge | Deciding overall task |
| Model Orchestrator | Model/provider selection | Tool execution |
| Tool Dispatcher | Tool selection/execution | Natural-language interpretation |
| Tool | One capability | Global orchestration |
| Kernel | Lifecycle/infrastructure | Intelligence decisions |
| EventBus | Communication | Business logic |

## 10. Roadmap Recovered From Project Notes

### NOW / FOUNDATION

- Conversation mode

- Persistent short-term memory

- Intent Router

- Tool Dispatcher

- Async streaming TTS

- HUD state machine

### NEXT / AWARENESS

- Desktop observers

- Clipboard / files / process awareness

- Screenshot + vision pipeline

- Tool permissions / authority

### LATER / ADVANCED

- Multi-agent hierarchy

- Background agents

- Multi-machine sidecars

- Persistent daemon/server architecture

## 11. Longer Version Roadmap

A.S.T.A. v1.0 — Foundation

Goal: Stable, modular architecture

- AI conversation

- Speech

- HUD

- Face recognition

- Telemetry

- Plugin architecture

- Memory

- Logging

A.S.T.A. v2.0 — Cognitive

Goal: More autonomous intelligence

- Long-term memory

- Task planning

- Self-improvement suggestions

- Autonomous tool selection

- Project awareness

A.S.T.A. v3.0 — Operating System

Goal: Control and awareness of the desktop environment

- Window automation

- Desktop overlay

- Notifications

- Workflow automation

- Calendar

- Email

- Browser control

A.S.T.A. v4.0

Goal: Genuine AI operating environment

## 12. Recovered GitHub Development Timeline

| Date | Recovered milestone |
| --- | --- |
| 29 Aug 2026 | Project notes/README established the Python kernel + EventBus direction. |
| 30 Aug 2026 | Intent routing introduced in AIModule. |
| 31 Aug 2026 | IntentRouter refactored to return IntentResult objects and improve entity extraction. |
| 1 Sep 2026 | Architecture diagram for Python kernel/EventBus added to README. |
| 2 Sep 2026 | Snake game experiments added, including alternate HTML/JS implementations. |
| 3 Sep 2026 | README and Snake structure refactored for clarity/functionality. |
| 5 Sep 2026 | Core contracts added/refined for IntentResult, MemoryRecord, ModelRequest, and Tool. |
| 6 Sep 2026 | Tool architecture introduced: Tool, ToolDispatcher, ToolRegistry, AuthorityPolicy, EchoTool; AI engine model updated to nvidia/nemotron-3-nano-4b. |
| 6 Sep 2026 | TFLite model files added to .gitignore. |

## 13. Current Repository Snapshot

GitHub repository: DEVpk108/Asta

- Default branch: main

- Public repository

- Current code-search index is available

- Core directory currently contains Kernel, EventBus, IntentRouter, Module, contracts, and tools.

- The repository also contains an AI package with AIModule and LM Studio engine code.

## 14. What Is Already Done vs. What Still Needs Work

| Area | Recovered status | Next focus |
| --- | --- | --- |
| Kernel/EventBus | Working foundation | Stabilize contracts + lifecycle tests |
| IntentRouter | Implemented | Improve classifier boundaries, confidence, and fallback behavior |
| Local LLM | Integrated with LM Studio | Fix/measure endpoint path and optimize TTFT |
| Streaming TTS | Partially present | Finish robust async sentence-to-speech pipeline |
| Memory | Contracts/architecture defined; full manager pending | Implement persistent short-term + retrieval |
| Tools | Architecture introduced | Connect real capabilities through dispatcher + authority |
| Authority | Policy layer introduced | Define risk model and approval behavior |
| HUD | Not yet complete | Implement state machine: idle/listening/thinking/speaking |
| Desktop awareness | Planned | Observers + clipboard/files/process awareness |
| Vision | Planned/early components exist | Screenshot ingestion + vision reasoning pipeline |
| Planner | Conceptual | Add only after tool/intent/memory boundaries are stable |

## 15. The Immediate Engineering Direction

Do not jump straight to multi-agent behavior. The recovered roadmap consistently places Intelligence Infrastructure before advanced autonomy.

- Finish and stabilize IntentRouter → ToolDispatcher boundary.

- Define a clean Tool contract and implement one or two real low-risk tools end-to-end.

- Finish persistent short-term memory and then add vector retrieval/context ranking.

- Correct and benchmark the LM Studio endpoint path; track TTFT consistently.

- Complete async sentence streaming into TTS.

- Implement HUD state machine only after event names and state transitions are stable.

- Add authority/risk levels before exposing powerful filesystem/process/system tools.

- Add tests around contracts and event-driven flows before adding complexity.

| Continuation rule<br>Architecture first, capabilities second. A.S.T.A. should stay modular: Kernel owns infrastructure, Intent handles understanding, Memory handles retrieval/storage, the model layer handles generation/model selection, Tools implement capabilities, ToolDispatcher governs execution, and EventBus connects components. |
| --- |

## 16. Important Project Philosophy

- A.S.T.A. should be local-first, not cloud-dependent by default.

- The LLM should not directly own the operating system.

- Every powerful action should pass through explicit tool and authority boundaries.

- The architecture must allow models to be swapped without rewriting the whole system.

- Memory should be selective and retrieval-driven rather than stuffing the complete history into every prompt.

- Performance matters: low latency, streaming, and efficient context allocation are first-class design goals.

- Build vertical slices that actually work before building speculative abstractions.

## 17. Project Continuation Protocol

This section defines the standard handoff record for continuing A.S.T.A. development after a project pause, context loss, or change of development environment.

A.S.T.A. CONTINUATION CHECKPOINT

A.S.T.A. is a local-first personal agentic AI / Cognitive OS. The repository is DEVpk108/Asta. The current architecture is Kernel + EventBus + modules + contracts + IntentRouter + AI engine + tool system. Do not redesign the entire system unless evidence requires it. Before extending the system, inspect the current repository state, continue from the Immediate Engineering Direction section, preserve component ownership boundaries, prefer small testable changes, and commit each meaningful architectural step.

## 18. Source / Confidence Notes

This record deliberately separates recovered facts from assumptions. The repository-backed sections are based on the current GitHub state and commit history available in this session. The broader project vision, roadmap, context-management strategy, component ownership rules, and some implementation intentions come from the additional project notes provided by the developer. Where a note describes something as planned, this document keeps it as planned rather than treating it as finished.

A.S.T.A. Project Memory • Updated 7 September 2026
