# asta roadmap

Converted reference from `asta_roadmap.docx`. The original document is retained.

Phase 1 — Interaction Core ✅

Wake words
Whisper
VAD
Conversation mode
Streaming LLM
Streaming sentence TTS
EventBus
LM Studio

Phase 2 — Intelligence Infrastructure ← NEXT

Intent Router
Tool Dispatcher
Persistent Memory Manager
Context Manager

Phase 3 — Capabilities

Filesystem
Applications
Browser
Screenshot
System controls
Notifications
OS awareness

Phase 4 — HUD

HUD v1:

Idle

Listening

Thinking

Speaking

Then HUD v2 can show:

Conversation

Memory

Tasks

Tools

System status

Phase 5 — advanced ASTA

Planner
Task execution
Background agents
Barge-in/interruption
Observability/telemetry
Multi-device sidecars

| Component | Owns | Does NOT own |
| --- | --- | --- |
| Intent Engine | understanding request | model selection, execution |
| Memory | storing/retrieving knowledge | deciding overall task |
| Model Orchestrator | model/provider selection | tool execution |
| Tool Dispatcher | tool selection/execution | natural-language interpretation |
| Tool | one capability | global orchestration |
| Kernel | lifecycle/infrastructure | intelligence decisions |
| EventBus | communication | business logic |

Problem 1: Wrong API Endpoint (Line 19-20)

You're using:

python

1

But LM Studio's OpenAI-compatible API endpoint is:

1

The /api/v1/chat endpoint is LM Studio's native API, which has different behavior and overhead compared to the OpenAI-compatible endpoint that LM Studio's built-in chat uses internally.
