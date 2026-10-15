# features

Converted reference from `features.docx`. The original document is retained.

Context Management

For the context management we can use this method:

Instead of loading up all the context and the old chat we can store the context in a Vector database and access only the info we need by searching it in the Vector DB.

Prompt ranking system With Dynamic Context Memory allocation

Prompts will be divided on the level of seriousness of the prompt and the context memory will be automatically changed according to that rank so that we can get high tok/s with less Generation time and memory for non-serious

Like  :- For prompt like hello it would be ranked as low

Then there will be Medium and High for reasoning and coding prompts.

NOW

────

1. Conversation mode

2. Persistent short-term memory

3. Intent Router

4. Tool Dispatcher

5. Async streaming TTS

6. HUD state machine

NEXT

────

7. Desktop observers

8. Clipboard/files/process awareness

9. Screenshot + vision pipeline

10. Tool permissions / authority

LATER

────

11. Multi-agent hierarchy

12. Background agents

13. Multi-machine sidecars

14. Persistent daemon/server architectur

Architecture

ASTA KERNEL

│

┌──────┴──────┐

│                                  │

EventBus       StateManager

│

┌─────────|───┼──────────────────┐

│                              │                                               │

Voice                      AI                                           Tools

│                              │                                               │

│                     Intent Router                              │

│                                │                                              │

│                  ┌────┴────┐                                 │

│                  │                        │                                  │

│             direct               LLM                               │

│                  │                         │                                 │

└───────┴─────────┴─────────────┘

│

┌─────|────────────────────────────┐

│              │                    │                                              │

Wake         VAD         Moonshine                         TTS

Word                               STT
