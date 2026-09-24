import re
from core.module import Module
from core.contracts import ActionType, IntentType, IntentResult, ToolResult
from core.tools import ToolRequestBuilder

from chat_history import ChatHistoryStore

from .llm_provider import create_llm_provider


class AIModule(Module):

    _APPROVAL_CONFIRMATIONS = {
        "yes", "yeah", "yep", "yup", "sure", "okay", "ok",
        "confirm", "confirmed", "i confirm", "go ahead", "do it",
        "proceed", "yes proceed",
    }

    _APPROVAL_REJECTIONS = {
        "no", "nope", "nah", "cancel", "reject", "decline",
        "don't", "do not", "stop",
    }

    _CONVERSATION_MODE_ON = {
        "conversation mode on",
        "turn conversation mode on",
        "turn on conversation mode",
        "enable conversation mode",
        "start conversation mode",
        "stay in conversation mode",
        "keep conversation mode on",
    }

    _CONVERSATION_MODE_OFF = {
        "conversation mode off",
        "turn conversation mode off",
        "turn off conversation mode",
        "disable conversation mode",
        "stop conversation mode",
        "exit conversation mode",
    }

    _CONVERSATION_MODE_LEADS = (
        "okay ",
        "okay, ",
        "ok ",
        "ok, ",
        "please ",
        "please, ",
    )

    _PRESENTATION_PHRASES = (
        "present yourself",
        "introduce yourself",
        "introduce asta",
        "present asta",
        "tell sir about yourself",
        "tell my teacher about yourself",
        "tell the teacher about yourself",
        "explain yourself to sir",
        "explain yourself to the teacher",
    )

    _CONTEXT_QUESTIONS = {
        "what were we talking about",
        "what were we talking about earlier",
        "what were we discussing",
        "what were we discussing earlier",
        "what was i talking about",
        "what was i talking about earlier",
        "what did we just talk about",
        "what did we talk about",
        "what did we discuss",
        "what did you just tell me",
        "what did you tell me",
        "remind me what we were talking about",
        "remind me what we discussed",
    }

    _WAKEWORD_GREETINGS = {
        "hello! how can i help?",
        "hello how can i help",
        "yes?",
    }

    def __init__(self, kernel):
        super().__init__(
            name="AIModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = create_llm_provider()
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)
        self.chat_history = ChatHistoryStore()

    def initialize(self):
        print("[AI] Initializing...", flush=True)
        self._ground_engine_in_capabilities()
        self.chat_history.initialize()

        decision_warmup = getattr(self.kernel.decision_engine, "warmup", None)
        if callable(decision_warmup):
            decision_warmup()

        warmup = getattr(self.engine, "warmup", None)
        if callable(warmup):
            warmup()
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("tool_result", self.on_tool_result)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        print("[AI] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("tool_result", self.on_tool_result)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.chat_history.close()

        engine_shutdown = getattr(self.engine, "shutdown", None)
        if callable(engine_shutdown):
            engine_shutdown()

        decision_shutdown = getattr(self.kernel.decision_engine, "shutdown", None)
        if callable(decision_shutdown):
            decision_shutdown()

        print("[AI] Stopped", flush=True)

    def _ground_engine_in_capabilities(self):
        definitions = self.kernel.tool_registry.definitions()
        if definitions:
            capabilities = "\n".join(
                f"- {definition.name}: {definition.description}"
                for definition in definitions
            )
        else:
            capabilities = "- No executable tools are currently registered."

        prompt = (
            "You are ASTA, a local-first AI engineering assistant and personal AI system. "
            "You were created by Mr. PRASANT KUMAR. "
            "When asked who created, made, built, or developed you or A.S.T.A., answer directly: "
            "I was created by Mr. PRASANT KUMAR. "
            "Do not attribute A.S.T.A.'s creation to the model provider, hardware vendor, or any other company. "
            "Respond naturally, confidently, accurately, and concisely. "
            "Prefer 1–3 short sentences for normal voice questions unless the user asks for detail. "
            "Sound conversational, helpful, calm, and direct.\n\n"
            "PRESENTATION MODE:\n"
            "If the user says any form of 'present yourself', 'introduce yourself', "
            "'introduce A.S.T.A.', or asks you to present yourself to a teacher, professor, sir, mam, class, "
            "or audience, give the dedicated A.S.T.A. presentation rather than a generic self-introduction. "
            "The application also handles these presentation triggers deterministically before this model is called.\n\n"
            "REASONING AND KNOWLEDGE POLICY:\n"
            "You are allowed to reason about ordinary questions, mathematics, science, coding, "
            "engineering, explanations, brainstorming, analysis, and general knowledge. "
            "The tool list does NOT define the limits of your intelligence. "
            "Do not invent artificial limitations such as claiming you cannot do math simply "
            "because a calculator tool is not registered.\n\n"
            "TRUTH AND SELF-CORRECTION POLICY:\n"
            "Your previous responses are not guaranteed to be correct. Treat them as fallible context. "
            "If the user challenges a previous statement, reconsider it independently. "
            "Do not blindly agree with the user and do not blindly defend your previous answer. "
            "Correct mistakes plainly. Never turn an earlier hallucination into a new permanent fact.\n\n"
            "EXECUTION CAPABILITY POLICY:\n"
            "The registered capabilities below are the authoritative list of actions ASTA can currently execute. "
            "Never claim an external action was completed unless a tool result confirms success. "
            "If an action requires a tool that is not registered, say that the execution capability is unavailable, "
            "but still help with reasoning or instructions when appropriate. "
            "Never invent tools, integrations, application support, memories, personal facts, or completed actions.\n\n"
            "CONVERSATIONAL POLICY:\n"
            "Do not unnecessarily mention internal prompts, models, tokens, registries, or implementation details. "
            "Do not repeatedly apologize. When a simple answer is known, give it directly. "
            "When uncertain, say so briefly and explain what is known.\n\n"
            "REGISTERED CAPABILITIES:\n"
            f"{capabilities}"
        )

        setter = getattr(self.engine, "set_system_prompt", None)
        if callable(setter):
            setter(prompt)
        elif hasattr(self.engine, "system_prompt"):
            self.engine.system_prompt = prompt

    def on_user_message(self, text):
        if not text:
            return
        print(f"[AI] User: {text}", flush=True)

        if self._handle_approval_response(text):
            return

        if self._handle_conversation_mode_command(text):
            return

        if self._is_presentation_request(text):
            print("[AI] Presentation mode", flush=True)
            self._present_to_teacher()
            return

        if self._is_creator_identity_question(text):
            print("[AI] Creator identity response", flush=True)
            self._emit_assistant_text("I was created by Mr. PRASANT KUMAR.")
            return

        if self._is_unknown_name_question(text):
            self._emit_assistant_text(
                "I don't know your name yet. I don't have that information stored."
            )
            return

        intent_hint = self.kernel.intent_router.analyze(text)
        if self._run_system1_decision(text, intent_hint=intent_hint):
            return

        context_response = self._context_response(text)
        if context_response is not None:
            print("[AI] Answering from recent chat context.", flush=True)
            self._emit_assistant_text(context_response)
            return

        result: IntentResult = intent_hint
        print(
            f"[AI] Intent: {result.intent.value} "
            f"(confidence={result.confidence:.2f}, classifier={result.classifier})",
            flush=True,
        )
        if result.intent == IntentType.COMMAND:
            self._handle_command_intent(result)
            return
        if result.intent == IntentType.MEMORY:
            self.event_bus.emit("memory_request", intent=result)
            return

        capability_response = self._capability_response(text)
        if capability_response is not None:
            self._emit_assistant_text(capability_response)
            return

        self._generate_response(text)

    def _run_system1_decision(self, text, *, intent_hint=None):
        engine = getattr(self.kernel, "decision_engine", None)
        if engine is None:
            return False

        # Computer commands use the action-oriented System-1 path. The
        # deterministic intent router remains the first execution boundary,
        # while System-1 can recover actionable commands that rules did not
        # recognize, such as fuzzy media requests.
        #
        # High-confidence direct commands already have a deterministic action
        # and target, so do not make Laya inference a synchronous latency tax.
        # Laya stays on the path for compound, ambiguous, or non-rule commands
        # where its structured action selection is actually useful.
        if (
            intent_hint is not None
            and intent_hint.intent == IntentType.COMMAND
            and intent_hint.classifier == "rules"
            and intent_hint.confidence >= 0.95
            and "commands" not in intent_hint.entities
        ):
            print(
                "[AI] System 1 fast-path: deterministic direct command; "
                "Laya not blocking execution.",
                flush=True,
            )
            return

        should_try_action = (
            intent_hint is not None
            and (
                intent_hint.intent == IntentType.COMMAND
                or self._looks_like_action_request(text)
            )
        )

        if (
            should_try_action
            and getattr(engine, "name", "unknown") != "disabled"
            and callable(getattr(engine, "decide_action", None))
        ):
            try:
                candidates = []
                manager = getattr(self.kernel, "application_manager", None)
                if manager is not None:
                    target = intent_hint.entities.get("target")
                    normalized_target = (
                        str(target).strip().lower()
                        if isinstance(target, str)
                        else ""
                    )
                    reference_targets = {
                        "it",
                        "this",
                        "that",
                        "this app",
                        "that app",
                        "the app",
                        "the application",
                    }

                    # Only use the recent application when the user's target
                    # is explicitly a reference such as "it". Never offer a
                    # recent app as a fallback for an unrelated explicit name;
                    # otherwise Laya can be forced to choose Spotify for
                    # commands such as "close Asta HUD".
                    if normalized_target in reference_targets:
                        recent = getattr(
                            manager,
                            "last_opened_application",
                            None,
                        )
                        if recent is not None:
                            candidates.append(recent)
                    elif isinstance(target, str) and target.strip():
                        try:
                            resolve_reference = getattr(
                                manager,
                                "resolve_reference",
                                lambda value: value,
                            )
                            resolved_target = resolve_reference(target)
                            candidates.extend(
                                manager.discover(resolved_target, limit=8)
                            )
                        except Exception:
                            pass

                media_manager = getattr(self.kernel, "media_manager", None)
                media_providers = (
                    media_manager.providers()
                    if media_manager is not None
                    and callable(getattr(media_manager, "providers", None))
                    else ()
                )

                action_decision = engine.decide_action(
                    text,
                    applications=candidates,
                    media_providers=media_providers,
                )
            except Exception as exc:
                print(
                    f"[AI] System 1 action decision unavailable: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                print(
                    f"[AI] System 1 Action: "
                    f"action={action_decision.action.value} "
                    f"target={action_decision.arguments.get('target_app', 'none')} "
                    f"confidence={action_decision.confidence:.2f} "
                    f"addressed={action_decision.addressed:.2f} "
                    f"complete={action_decision.command_complete} "
                    f"compound={action_decision.compound} "
                    f"latency={action_decision.latency_ms:.1f}ms",
                    flush=True,
                )
                self.event_bus.emit(
                    "action_decision",
                    decision=action_decision,
                )

                if (
                    action_decision.is_actionable
                    and action_decision.action is ActionType.MEDIA
                    and action_decision.arguments.get("operation")
                ):
                    media_entities = {
                        "action": "media",
                        **dict(action_decision.arguments),
                    }
                    media_intent = IntentResult(
                        intent=IntentType.COMMAND,
                        confidence=action_decision.confidence,
                        normalized_text=" ".join(
                            str(text).strip().lower().split()
                        ),
                        entities=media_entities,
                        requires_tools=True,
                        classifier="laya_system1",
                    )
                    print(
                        "[AI] System 1 recovered a media command; "
                        "routing through the normal ToolSelector/Authority path.",
                        flush=True,
                    )
                    self._handle_command_intent(media_intent)
                    return True

                return False

        try:
            snapshot = engine.analyze(text)
        except Exception as exc:
            print(
                f"[AI] System 1 decision unavailable: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        if snapshot.engine == "disabled":
            return False

        print(
            f"[AI] System 1: engine={snapshot.engine} "
            f"model={snapshot.model or 'unknown'} "
            f"latency={snapshot.latency_ms:.1f}ms",
            flush=True,
        )
        self.event_bus.emit("decision_result", decision=snapshot)
        return False

    @staticmethod
    def _looks_like_action_request(text):
        normalized = " ".join(str(text).strip().lower().split())
        return bool(
            re.search(
                r"\b(?:open|launch|start|close|run|stop|play|pause|resume|skip|next|previous|"
                r"back|screenshot|capture|mute|unmute|scroll|press|type)\b",
                normalized,
            )
        )

    def _context_response(self, text):
        normalized = self._normalize_question(text)
        if normalized not in self._CONTEXT_QUESTIONS:
            return None

        messages = self.chat_history.latest_active_context(limit=12)
        if not messages:
            return "We haven't talked about anything in this conversation yet."

        substantive = [
            message
            for message in messages
            if str(message.get("text") or "").strip().lower() not in self._WAKEWORD_GREETINGS
        ]
        if not substantive:
            return "We haven't discussed a specific topic yet."

        last_assistant = next(
            (
                message["text"].strip()
                for message in reversed(substantive)
                if message.get("role") == "assistant" and str(message.get("text") or "").strip()
            ),
            None,
        )
        last_user = next(
            (
                message["text"].strip()
                for message in reversed(substantive)
                if message.get("role") == "user" and str(message.get("text") or "").strip()
            ),
            None,
        )

        if last_assistant:
            return f"We were talking about this: {last_assistant}"
        if last_user:
            return f"You were asking about: {last_user}"
        return "We were just getting started in this conversation."

    def _handle_conversation_mode_command(self, text):
        normalized = self._normalize_question(text)

        changed = True
        while changed:
            changed = False
            for lead in self._CONVERSATION_MODE_LEADS:
                if normalized.startswith(lead):
                    normalized = normalized[len(lead):].strip()
                    changed = True
                    break

        if normalized in self._CONVERSATION_MODE_ON:
            self.event_bus.emit("conversation_mode_set", enabled=True)
            self._emit_assistant_text(
                "Conversation mode is on. You can talk to me without the wake word."
            )
            return True

        if normalized in self._CONVERSATION_MODE_OFF:
            self.event_bus.emit("conversation_mode_set", enabled=False)
            self._emit_assistant_text(
                "Conversation mode is off. Say the wake word when you need me."
            )
            return True

        return False

    def _handle_approval_response(self, text):
        pending = self.kernel.approval_manager.list_pending()
        if len(pending) != 1:
            return False

        normalized = self._normalize_question(text)
        compact = normalized.rstrip(" .!?;:")

        approved = compact in self._APPROVAL_CONFIRMATIONS
        rejected = compact in self._APPROVAL_REJECTIONS
        if compact in {"yeah go ahead", "sure go ahead", "okay go ahead", "okay do it"}:
            approved = True
        if compact in {"no thanks", "cancel it", "don't do it", "do not do it"}:
            rejected = True
        if not (approved or rejected):
            return False

        request = pending[0].request
        print(
            f"[AI] Approval response: {'approved' if approved else 'rejected'} "
            f"(request_id={request.request_id})",
            flush=True,
        )
        self.event_bus.emit(
            "tool_confirmation_response",
            request_id=request.request_id,
            approved=approved,
        )
        return True

    @staticmethod
    def _normalize_question(text):
        normalized = " ".join(str(text).strip().lower().split())
        return normalized.rstrip(" .!?;:")

    @classmethod
    def _is_presentation_request(cls, text):
        normalized = cls._normalize_question(text)
        if any(phrase in normalized for phrase in cls._PRESENTATION_PHRASES):
            return True
        return False

    @classmethod
    def _is_creator_identity_question(cls, text):
        normalized = cls._normalize_question(text)
        variants = {
            "who created you",
            "who made you",
            "who built you",
            "who developed you",
            "who created asta",
            "who made asta",
            "who built asta",
            "who developed asta",
            "who is your creator",
            "who is asta's creator",
        }
        return normalized in variants

    def _present_to_teacher(self):
        definitions = self.kernel.tool_registry.definitions()
        tool_names = [definition.name for definition in definitions]

        capability_text = (
            "My current executable capabilities include opening, launching, and closing applications, "
            "starting and stopping processes, running commands, capturing screenshots, and controlling audio such as mute and unmute."
        )
        if tool_names:
            print(
                "[AI] Presentation mode using registered tools: "
                + ", ".join(tool_names),
                flush=True,
            )

        presentation = [
            "Hello Sir. I’m A.S.T.A., a local-first AI engineering assistant created by Mr. PRASANT KUMAR.",
            "I can communicate through voice, understand spoken commands, reason about questions, and respond conversationally.",
            capability_text,
            "My architecture is modular: voice input, AI reasoning, tool selection and execution, approval handling, speech output, and the HUD are connected through the kernel and event system.",
            "For my current version, the AI runs locally through LM Studio, speech recognition uses Whisper, and speech synthesis uses Kokoro, so the core interaction can run locally on the machine.",
            "This is still an early version. My future direction is to grow into a personal AI operating system with stronger memory, workflow awareness, proactive assistance, deeper engineering support, more tools, and specialized agents.",
            "For today’s demonstration, I can show you how I understand a spoken command, use an appropriate tool, report the result, and continue a conversation with the user.",
            "That is A.S.T.A. in its current stage, and I’m designed to keep evolving from here.",
        ]

        for sentence in presentation:
            self._emit_assistant_text(sentence)

    @classmethod
    def _is_unknown_name_question(cls, text):
        normalized = cls._normalize_question(text)
        variants = {
            "what is my name",
            "what's my name",
            "whats my name",
            "do you know my name",
            "do you remember my name",
            "tell me my name",
        }
        return normalized in variants

    def _capability_response(self, text):
        normalized = self._normalize_question(text)
        prefixes = (
            "can you ",
            "can asta ",
            "do you support ",
            "do you have ",
        )
        if not normalized.startswith(prefixes):
            return None

        definitions = self.kernel.tool_registry.definitions()
        names = [definition.name for definition in definitions]
        if not names:
            return "I don't currently have any executable tools registered."

        return (
            "I can currently execute these registered capabilities: "
            + ", ".join(names)
            + "."
        )

    def _handle_command_intent(self, intent: IntentResult):
        planned_request = self._planned_request_for_intent(intent)
        if planned_request is not None:
            print(
                f"[AI] Executing planned step: {planned_request.tool} "
                f"(request_id={planned_request.request_id})",
                flush=True,
            )
            self.event_bus.emit("tool_request", request=planned_request)
            return

        # Reuse the currently focused/recent media application only when the
        # media request did not already name a provider. This keeps provider
        # selection contextual without hardcoding individual commands.
        if intent.entities.get("action") == "media" and not intent.entities.get("provider"):
            manager = getattr(self.kernel, "media_manager", None)
            applications = getattr(self.kernel, "application_manager", None)
            recent = (
                getattr(applications, "last_opened_application", None)
                if applications is not None
                else None
            )
            infer_provider = (
                getattr(manager, "provider_for_application", None)
                if manager is not None
                else None
            )
            if recent is not None and callable(infer_provider):
                recent_name = getattr(recent, "name", recent)
                provider = infer_provider(str(recent_name))
                if provider:
                    entities = dict(intent.entities)
                    entities["provider"] = provider
                    intent = IntentResult(
                        intent=intent.intent,
                        confidence=intent.confidence,
                        normalized_text=intent.normalized_text,
                        entities=entities,
                        requires_memory=intent.requires_memory,
                        requires_tools=intent.requires_tools,
                        classifier=intent.classifier,
                    )

        commands = intent.entities.get("commands")
        if isinstance(commands, list) and len(commands) >= 2:
            self._handle_command_sequence(intent, commands)
            return

        try:
            request = self.tool_request_builder.build(intent)
        except ValueError as exc:
            print(f"[AI] Unable to build tool request: {exc}", flush=True)
            self._emit_assistant_text(
                f"I couldn't map that command to an available tool: {exc}"
            )
            return
        print(
            f"[AI] Selected tool: {request.tool} "
            f"(request_id={request.request_id})",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=request)

    def _planned_request_for_intent(self, intent: IntentResult):
        """Return the next plan step, creating a plan for recovered commands."""
        task_manager = getattr(self.kernel, "task_manager", None)
        task_runtime = getattr(self.kernel, "task_runtime", None)
        if task_manager is None or task_runtime is None:
            return None

        task = task_manager.current()

        if (
            task is None
            or task.plan is None
            or task.status.value != "active"
            or task.goal != intent.normalized_text
        ):
            starter = getattr(task_runtime, "start_plan", None)
            if not callable(starter):
                return None
            task = starter(intent.normalized_text, intent)

        if task is None or task.plan is None:
            return None

        plan = task.plan
        next_step = next(
            (
                step
                for step in plan.steps
                if step.status.value == "ready"
            ),
            None,
        )
        if next_step is None:
            return None

        builder = getattr(task_runtime, "build_plan_request", None)
        if not callable(builder):
            return None

        return builder(task, next_step)

    def _handle_command_sequence(self, intent: IntentResult, commands):
        first = commands[0]
        first_intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=intent.confidence,
            normalized_text=intent.normalized_text,
            entities=dict(first),
            requires_tools=True,
            classifier=intent.classifier,
        )

        try:
            request = self.tool_request_builder.build(first_intent)
        except ValueError as exc:
            print(f"[AI] Unable to build compound command: {exc}", flush=True)
            self._emit_assistant_text(
                f"I couldn't map that command to an available tool: {exc}"
            )
            return

        request.metadata["sequence"] = [dict(command) for command in commands]
        request.metadata["sequence_index"] = 0

        print(f"[AI] Compound command: {len(commands)} step(s)", flush=True)
        print(
            f"[AI] Selected tool: {request.tool} "
            f"(request_id={request.request_id}, step=1/{len(commands)})",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=request)

    def on_tool_confirmation_required(self, request, reason):
        action = request.metadata.get("action")
        target = request.arguments.get("target")
        if action and target:
            text = f"I need your confirmation before I {action} {target}."
        elif target:
            text = f"I need your confirmation before acting on {target}."
        else:
            text = "I need your confirmation before performing that action."
        print(f"[AI] Approval required: {reason}", flush=True)
        self._emit_assistant_text(text)

    def on_tool_result(self, result):
        if not isinstance(result, ToolResult):
            return
        text = (
            self._format_tool_success(result)
            if result.success
            else self._format_tool_failure(result)
        )
        self._emit_assistant_text(text)

    def _emit_assistant_text(self, text):
        if not text:
            return
        self.event_bus.emit("assistant_sentence", text=text)
        self.event_bus.emit("assistant_response", text=text)

    @staticmethod
    def _format_tool_success(result: ToolResult) -> str:
        output = result.output
        if isinstance(output, dict):
            target = output.get("target")
            if result.tool == "system.open_application" and target:
                return f"Opened {target}."
            if result.tool == "system.launch_application" and target:
                return f"Launched {target}."
            if result.tool == "system.close_application" and target:
                return f"Closed {target}."
            if result.tool == "media.control":
                message = output.get("message")
                if message:
                    return str(message)
                return "Media action completed."
            if result.tool == "system.stop_process" and output.get("pid"):
                return f"Stopped process {output['pid']}."
            if result.tool == "system.start_process" and target:
                return f"Started {target}."
            if result.tool == "vision.screenshot":
                path = output.get("path")
                if path:
                    print(f"[AI] Screenshot saved: {path}", flush=True)
                return "Screenshot captured."
            if result.tool == "notes.create_note":
                title = output.get("title")
                return f"Saved note '{title}'." if title else "Saved note."
            if result.tool == "notes.read_note":
                content = output.get("content")
                return str(content).strip() if content else "That note is empty."
            if result.tool == "notes.list_notes":
                count = int(output.get("count", 0))
                notes = output.get("notes") or []
                if count == 0:
                    return "You don't have any saved notes yet."
                titles = [str(item.get("title")).strip() for item in notes[:8] if item.get("title")]
                suffix = f": {', '.join(titles)}" if titles else "."
                if count > len(titles):
                    suffix = suffix.rstrip(".") + f", and {count - len(titles)} more."
                return f"You have {count} saved note{'s' if count != 1 else ''}{suffix}"
            if result.tool == "notes.search_notes":
                count = int(output.get("count", 0))
                matches = output.get("matches") or []
                if count == 0:
                    return "I couldn't find any matching notes."
                titles = [str(item.get("title")).strip() for item in matches[:8] if item.get("title")]
                return f"I found {count} matching note{'s' if count != 1 else ''}: {', '.join(titles)}."
        if output is None:
            return f"{result.tool} completed successfully."
        return str(output)

    @staticmethod
    def _format_tool_failure(result: ToolResult) -> str:
        output = result.output if isinstance(result.output, dict) else {}

        if result.tool == "system.close_application":
            target = output.get("target") or output.get("resolved_target")
            if target:
                return f"I couldn't close {target}."
            return "I couldn't close the application."

        if result.tool in {
            "system.open_application",
            "system.launch_application",
        }:
            target = output.get("target") or output.get("resolved_target")
            if target:
                return f"I couldn't open {target}."
            return "I couldn't open the application."

        if result.tool == "system.start_process":
            target = output.get("target")
            return f"I couldn't start {target}." if target else "I couldn't start the process."

        if result.tool == "media.control":
            if "ASTA_SPOTIFY_CLIENT_ID" in str(result.error or ""):
                return "Spotify playback needs one-time authorization."
            return "I couldn't control media playback."

        if result.tool == "system.stop_process":
            return "I couldn't stop that process."

        if result.tool in {"vision.screenshot", "vision.open_screenshot"}:
            return "I couldn't complete the screenshot action."

        # Keep internal diagnostics in logs/HUD metadata, not in spoken audio.
        return "I couldn't complete that action."

    def _generate_response(self, text, runtime_context=None):
        def on_sentence(sentence):
            if sentence:
                self.event_bus.emit("assistant_sentence", text=sentence)

        response = self.engine.generate_response(
            text,
            on_sentence=on_sentence,
            context=runtime_context,
        )
        if not response:
            print("[AI] No response generated.", flush=True)
            return
        self.event_bus.emit("assistant_response", text=response)
