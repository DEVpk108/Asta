# ASTA Improvement Technical Documentation

Converted reference from `ASTA_Improvement_Technical_Documentation.docx`. The original document is retained.

A.S.T.A. Improvement Technical Documentation

Voice reliability, interruption, intent routing, tool execution, presentation mode, and runtime behavior

Project: A.S.T.A. (local-first AI engineering assistant)
Repository: DEVpk108/Asta
Baseline: main @ c43db9ddc3312566c4b766122e458f53494b5ba1
Documentation status: September 2026

| Purpose<br>This document records the engineering improvements made to A.S.T.A. during the conversational, voice, tool-runtime, and presentation hardening work. It explains not only what changed, but how the major features operate internally and how the modules communicate. |
| --- |

## 1. Executive Summary

The recent A.S.T.A. work moved the system from a basic voice assistant toward a more dependable event-driven local AI runtime. The focus was reliability: speech recognition should not hallucinate commands, speech should be interruptible, compound commands should be deterministic, tools should be selected from the registered capability set, the model should not claim actions that were never executed, and presentation behavior should be predictable.

The major implementation areas are:

- Wake-word confirmation and conversation lifecycle hardening.

- Whisper speech-to-text stabilization and hallucination filtering.

- A non-consuming microphone-ring-buffer barge-in path for interruptible speech.

- Short-response confirmation handling for high-risk tools.

- Rule-based intent routing and compound command parsing.

- Registry-backed deterministic tool selection and event-driven dispatch.

- Conversation grounding so the LLM knows what ASTA can actually execute.

- Deterministic creator identity and deterministic presentation mode.

- Single-worker TTS synthesis/playback with interruption support.

- Deterministic mixed-request handling so conversational answers can be followed by a real screenshot tool call.

## 2. Runtime Architecture After the Improvements

A.S.T.A. is organized around a Kernel and EventBus. Modules subscribe to events and do not need to call each other directly for most runtime interactions. This separates language understanding, authority, tool execution, voice input, speech output, and UI/HUD behavior.

Microphone -> WakeWord/VAD -> Whisper STT -> VoiceModule
                                      |
                                      v
                                  Kernel/EventBus
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                     AIModule                 SpeechModule
                         |                         |
                  IntentRouter                   Kokoro
                         |
                  ToolRequestBuilder
                         |
                   ToolSelector
                         |
                  ToolRuntimeModule
                         |
                  ToolDispatcher
                         |
                Concrete Tool (e.g. screenshot)
                         |
                         v
                    ToolResult -> EventBus -> AIModule/HUD

| Layer | Responsibility | Key components |
| --- | --- | --- |
| Voice input | Wake-word, conversation state, VAD collection, STT, barge-in | VoiceModule, WakeWordEngine, VADEngine, RecognitionEngine |
| Language understanding | Rule-based intent routing and deterministic special cases | IntentRouter, AIModule |
| LLM reasoning | Normal knowledge/conversation responses, context continuity | AIEngine via LM Studio |
| Tool planning | Map structured command intent to a registered capability | ToolRequestBuilder, ToolSelector, ToolRegistry |
| Tool execution | Authorization, dispatch, approval, result events | ToolRuntimeModule, ToolDispatcher, ApprovalManager |
| Speech output | Queueing, synthesis, playback, interruption | SpeechModule, KokoroEngine |
| Presentation/UI | HUD output and presentation flow | HUDModule, presentation runtime patch |

## 3. Voice Input and Conversation Lifecycle

### 3.1 Wake-word confirmation

A.S.T.A. listens for one of the registered wake words: hello_asta, hey_asta, or wake_up_asta. Detection uses a threshold of 0.35 and requires two consecutive confirmation frames. Even a strong score does not bypass the two-frame confirmation.

Wake-word score -> threshold check -> consecutive-frame confirmation -> conversation ACTIVE -> greeting -> command listener

The important architectural point is that wake-word detection starts a temporary conversation window. It does not permanently leave the microphone in free-form command mode unless manual conversation mode has been enabled.

### 3.2 Conversation mode

After a confirmed wake word, VoiceModule marks the conversation active and records the last interaction time. The default conversation timeout is 30 seconds. Manual conversation mode can also be turned on or off through deterministic commands.

- ACTIVE: commands can be spoken without repeating the wake word.

- INACTIVE: A.S.T.A. returns to wake-word listening.

- Manual mode bypasses the normal timeout behavior.

- Speech output temporarily prevents the normal command reader from consuming microphone audio.

## 4. Whisper STT Hardening

A major goal was reducing the chance that silence, residual audio, or speech context would produce invented Whisper text. The runtime therefore applies several independent safeguards.

| Protection | Implementation effect |
| --- | --- |
| No literal Whisper initial prompt | Avoids biasing transcription toward predefined phrases. |
| condition_on_previous_text=False | Prevents prior decoded text from steering later segments. |
| Silero VAD remains authoritative | Whisper is given already-segmented speech rather than deciding speech boundaries a second time. |
| Segment metrics | avg_logprob, no_speech_prob, and compression_ratio are inspected before accepting transcription. |
| Hallucination phrase filtering | Known non-speech / repetitive Whisper artifacts are rejected. |
| Duplicate transcript guard | Identical text repeated within a short window is rejected as likely residual audio. |
| Fresh buffers after wake word | Wake-word audio is not reused as the user command. |
| Fresh pre-roll / command buffer | Command collection begins from clean microphone state. |

The design intentionally treats STT as an untrusted input source. The rest of the system should only react to a transcript after it has passed the voice pipeline safeguards.

## 5. Barge-in / Speech Interrupt: How It Works at the Core

The interrupt feature is one of the most important reliability improvements. The core problem is that the same microphone stream cannot safely be consumed by both the normal VAD command listener and the speech-interrupt detector. The solution is to make interruption observation non-consuming.

### 5.1 The core design

The runtime patch replaces the original barge-in recognizer with a listener that observes the microphone ring buffer. It does not steal chunks from the normal voice/VAD command reader.

TTS speaking
   |
   +--> SpeechModule emits speech_started
            |
            +--> VoiceModule enables barge-in worker
                     |
                     +--> read microphone ring buffer
                     |      (no destructive consumption)
                     |
                     +--> keep recent ~2 seconds
                     |
                     +--> wait ~0.25 s between checks
                     |
                     +--> reject low-RMS silence
                     |
                     +--> Whisper transcribes buffer
                     |
                     +--> interrupt phrase matcher
                              |
                       +------+-------+
                       |              |
                      no             yes
                       |              |
                    continue     emit speech_interrupt
                                      |
                                      +--> SpeechModule drains queue
                                      +--> Kokoro playback stops
                                      +--> VoiceModule disables barge-in
                                      +--> optional tail becomes new user_message

### 5.2 Why the ring buffer matters

The user may say “okay, okay, stop” while speech is already playing. If the interrupt listener only listened to the newest tiny chunk, it could miss the beginning of the phrase. The patched implementation keeps up to two seconds of recent microphone samples, giving Whisper enough context to decode short interruption phrases reliably.

### 5.3 Audio gating

The barge listener checks that enough audio has accumulated and that the signal RMS is above a minimum threshold. This prevents repeated Whisper calls on silence.

max buffer = 2.0 s
minimum speech window = 0.55 s
check interval ~= 0.25 s
minimum RMS = 0.006

### 5.4 Interrupt phrase extraction

The matcher accepts explicit phrases including “okay stop”, “stop ASTA”, “stop speaking”, “stop presentation”, “hold on”, and a bare “stop”. It also handles Whisper variants where acknowledgement words appear before stop.

A critical safety behavior is avoiding false interruption triggers from ordinary sentences. For example, the base VoiceModule deliberately distinguishes “stop” from arbitrary text such as “please stop the process.”

### 5.5 Event flow after detection

- Barge worker recognizes an interrupt phrase.

- VoiceModule emits the speech_interrupt event.

- SpeechModule sets the interrupt event and drains queued speech items.

- The current Kokoro playback loop sees should_continue() become false.

- Playback stops without waiting for the full synthesized clip to finish.

- VoiceModule clears the microphone state and disables the barge listener.

- If the interrupt transcript had a meaningful tail after the interrupt phrase, that tail can continue as a new user message.

This is why the interrupt feature is not merely a keyword detector. It is a cross-module cancellation protocol coordinated through the EventBus.

## 6. Tool Confirmation and Short-Reply Listening

High-risk tools can require user approval before execution. During confirmation, normal VAD settings are temporarily relaxed because replies such as “yes”, “okay”, or “no” are short and quiet.

- VoiceModule marks _awaiting_confirmation=True when approval is requested.

- VAD thresholds are temporarily lowered and minimum speech duration is shortened.

- Whisper output is normalized to decision words where appropriate.

- AIModule maps confirmations such as yes, yeah, yep, yup, sure, okay, confirm, go ahead, and do it to approval.

- Rejection phrases such as no, nope, nah, cancel, and stop are mapped to rejection.

- ToolRuntimeModule resumes dispatch only when approved.

## 7. Deterministic Intent Routing and Compound Commands

A.S.T.A. does not ask the LLM to decide every computer action. The IntentRouter converts common spoken command patterns into structured IntentResult objects before tool execution.

| User phrase | Structured intent |
| --- | --- |
| Open Chrome | action=open, target=chrome |
| Take a screenshot | action=screenshot |
| Open WhatsApp and take screenshot | commands=[open WhatsApp, screenshot] |
| Close camera, close Chrome | commands=[close camera, close Chrome] |
| Open camera take screenshot | commands=[open camera, screenshot] |

Compound command parsing recognizes separators such as “and”, “then”, “and then”, “after that”, and comma-separated command boundaries. It also contains an explicit rule for implicit screenshot suffixes, which is why phrases like “open camera take screenshot” can be decomposed without an LLM.

## 8. Tool Selection, Runtime, and Authority

### 8.1 Registry-backed capability discovery

Tool definitions are registered before the Kernel starts. The ToolSelector scores registered capabilities against the structured action and the entities produced by IntentRouter. It prefers exact action metadata, schema compatibility, and required inputs.

IntentResult
   -> ToolRequestBuilder
      -> ToolSelector
         -> registered ToolDefinition
            -> ToolRequest
               -> ToolRuntimeModule
                  -> ToolDispatcher
                     -> ToolResult

### 8.2 Why this matters

The tool list is a real runtime capability boundary. The LLM system prompt is populated from the registry, and the execution policy explicitly tells the model not to claim that a tool ran unless a ToolResult confirms success. This separation prevents “tool hallucination” from becoming a computer action.

The registered runtime currently includes application control, process control, command execution, screenshots, and audio mute/unmute. The system prompt is generated from the registry rather than duplicating the capability list manually.

## 9. Mixed Conversational Request + Real Tool Execution

The recent validation exposed an important LLM failure mode: for “tell me a joke then take a screenshot”, the model initially produced natural-language text saying it would take a screenshot, but no ToolRequest was emitted. The runtime patch solves this by recognizing the mixed pattern before the LLM can invent an execution claim.

User: "Tell me a joke and then take a screenshot"
      |
      +--> mixed-request parser
              |
              +--> prompt = "Tell me a joke"
              |
              +--> normal LLM response
              |
              +--> explicit IntentResult(action=screenshot)
                      |
                      +--> ToolRequestBuilder
                              |
                              +--> vision.screenshot
                                      |
                                      +--> ToolRuntimeModule
                                      +--> actual PNG file
                                      +--> ToolResult
                                      +--> HUD/speech confirmation

This keeps reasoning and execution separate: the LLM answers the knowledge portion; deterministic application code owns the external action.

## 10. Deterministic Creator Identity

Creator identity is handled before normal LLM generation. This avoids dependence on model memory or prompt following for a stable identity fact.

The canonical identity is: “I was created by Mr. PRASANT KUMAR.” The runtime also recognizes natural variants such as who created you, who built ASTA, who is your creator, who developed ASTA, who is behind ASTA, and who is your developer. The runtime patch expanded the matching to cover conversational variants.

## 11. Presentation Mode

Presentation mode is deterministic. Recognized phrases such as “present yourself” and “introduce yourself” are handled by AIModule before normal model generation. The presentation is emitted as a fixed sequence of sentences so the demonstration is repeatable.

- Identity and creator statement.

- Voice and conversation capability statement.

- Current executable tool capability statement.

- Modular architecture statement.

- Local AI stack statement: LM Studio, Whisper, Kokoro.

- Long-term direction toward a personal AI operating system.

- Demonstration workflow explanation.

- Closing statement about evolution and future growth.

The later presentation speech patch also removed the shortened spoken script. Speech now receives the same presentation sentence stream that the HUD displays, so the voice and visual output remain aligned.

## 12. Speech / TTS Pipeline and Cancellation

SpeechModule uses one worker thread to own synthesis and playback. This avoids multiple sentences fighting over audio output.

assistant_sentence
      |
      v
 Speech queue
      |
 Speech worker
      |
 Kokoro synthesize
      |
 Kokoro play(chunks, should_continue)
      |
 speech_finished

The playback loop receives a cancellation predicate. When speech_interrupt is raised, that predicate becomes false, causing playback to terminate before the entire clip is played. The queue is also drained so old sentences do not continue after an interruption.

## 13. Conversational Grounding and Truth Policy

A.S.T.A. receives a system prompt that describes its identity, registered capabilities, reasoning policy, and execution-truth policy. The important rule is that a model response is not considered evidence that a computer action occurred. The ToolRuntimeModule and ToolResult are the evidence.

- Normal questions may still be answered by the LLM even when no tool exists.

- Lack of a calculator tool does not imply lack of mathematical reasoning ability.

- Previous responses are treated as fallible context.

- The model is instructed not to invent tools, integrations, memories, or completed actions.

- External execution is considered complete only after a successful ToolResult.

## 14. Important Events and Their Meaning

| Event | Purpose | Typical producer/consumer |
| --- | --- | --- |
| user_message | Normalized user input for AI processing | Voice/Text -> AIModule |
| assistant_sentence | Stream a spoken/displayable sentence | AIModule -> Speech/HUD/Voice state |
| assistant_response | Complete response notification | AIModule -> UI / consumers |
| speech_started | Marks active TTS playback | SpeechModule -> VoiceModule |
| speech_finished | Marks end of TTS playback | SpeechModule -> VoiceModule |
| speech_interrupt | Cancel current speech | VoiceModule -> SpeechModule |
| tool_request | Request an executable capability | AIModule -> ToolRuntimeModule |
| tool_result | Execution result | ToolRuntimeModule -> AIModule |
| tool_confirmation_required | User approval is needed | ToolRuntimeModule -> AI/Voice |
| tool_confirmation_response | User approved/rejected a pending action | AI/Voice -> ToolRuntimeModule |
| conversation_mode_set | Explicitly enable or disable conversation mode | AIModule -> VoiceModule |

## 15. Validation and Observed Runtime Behavior

The latest local validation run confirmed the most important recent changes:

- Wake-word confirmation worked and activated conversation mode.

- Presentation mode was triggered deterministically by “Okay, present yourself”.

- Barge-in successfully detected “stop.” while the long presentation was playing.

- Kokoro playback stopped immediately after the interrupt event.

- The mixed request “Tell me a joke and then take a screenshot” generated the joke first, then executed the real vision.screenshot tool.

- The screenshot was actually written to runtime/screenshots as a PNG.

- The HUD reported “Screenshot captured.” only after tool execution.

- A normal “open ...” command still routes through system.open_application.

| Current discovered limitation<br>“Open the screenshot” is currently interpreted as an application-opening command with target “the screenshot”. There is not yet a dedicated capability for opening the most recent generated screenshot. This is a separate feature gap from screenshot capture itself. |
| --- |

## 16. Non-Blocking Runtime Warnings Observed

- Hugging Face Hub unauthenticated-request warning: affects download rate limits, not A.S.T.A. runtime correctness.

- PyTorch RNN dropout warning: indicates dropout=0.2 with num_layers=1; this is a model/library configuration warning.

- PyTorch weight_norm FutureWarning: indicates a future API migration from torch.nn.utils.weight_norm to the parametrizations API.

## 17. Engineering Lessons from the Improvements

- Do not let the LLM be the sole authority for computer actions.

- Treat speech recognition as noisy input and add deterministic validation around it.

- Use non-consuming observation paths for cross-cutting listeners such as barge-in.

- Keep capability discovery in the registry and selection layer instead of hard-coding it into the LLM prompt alone.

- Use events for cancellation and lifecycle coordination between independent modules.

- Separate conversational reasoning from deterministic external side effects.

- When a behavior must be reproducible for a demo, implement that behavior deterministically instead of relying on model generation.

- Validate end-to-end behavior with terminal logs, not only unit-level assumptions.

## 18. Source Map

| Source | Role |
| --- | --- |
| ai/ai_module.py | Identity, deterministic presentation, tool-request creation, tool-result formatting, LLM grounding. |
| ai/openai_engine.py | LM Studio request/streaming logic, conversation response chain, token/timing metrics. |
| ai/runtime_patch.py | Mixed conversational response + screenshot handling and expanded creator matching on the current feature branch. |
| core/intent_router.py | Rule-based command parsing, compound command segmentation, implicit screenshot suffix handling. |
| core/tools/request_builder.py | Intent -> ToolRequest conversion. |
| core/tools/selector.py | Registry-backed tool capability scoring and selection. |
| core/tools/module.py | Tool request dispatch, approvals, ToolResult emission, sequence continuation. |
| core/tools/builtin.py | Concrete system, screenshot, process, and audio tools. |
| voice/voice_module.py | Conversation lifecycle, VAD command collection, confirmation mode, speech lifecycle. |
| voice/interrupt_patch.py | Ring-buffer barge-in implementation and interrupt phrase matching. |
| speech/speech_module.py | Speech queue, single worker, presentation speech behavior, interrupt draining. |
| speech/kokoro_engine.py | Kokoro synthesis and interruptible playback. |
| main.py | Kernel construction, tool registration, module registration and runtime startup. |
| vision/screenshot_backend.py | Windows full-desktop screenshot capture to PNG and cross-platform fallback. |
| tests/test_intent_router.py | Compound command and screenshot intent regressions. |
| tests/test_ai_grounding.py | Capability prompt and non-invented capability behavior. |
| tests/test_builtin_tools.py | Screenshot and built-in tool behavior. |
| tests/test_tool_runtime.py | Event-driven tool runtime dispatch regression. |
| tests/test_runtime_patches.py | Mixed response + screenshot, creator matching, and presentation patch regressions. |

## 19. Current Development State

The recent improvement work is organized around a feature branch/PR workflow so behavior can be validated before merging into main. The most recently validated branch was fix/voice-tool-execution, which contains the mixed-request execution and presentation synchronization fixes.

Next planned capability gap identified by validation: a dedicated “open latest screenshot” action rather than treating the screenshot filename as an application target.
