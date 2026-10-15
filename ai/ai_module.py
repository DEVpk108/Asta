import os
import re
import time

from core.quick_answers import quick_math, unsupported_file_action
from core.agent import intent_resolver, project_followup, work_awareness
from core.agent.project_conversation import ProjectConversation
from difflib import SequenceMatcher
from datetime import datetime, timezone
from core.module import Module
from core.contracts import ActionType, IntentType, IntentResult, ToolRequest, ToolResult
from core.tools import ToolRequestBuilder
from core.approval_language import classify_approval, normalize_confirmation_transcript
from core.context_builder import ContextBuilder

from chat_history import ChatHistoryStore, shared_history

from .post_response_capture import PostResponseCapture, extract_prompt
from .message_shortcuts import handle_shortcut, _is_creator_identity_question as is_creator_question
from .response_formatter import (format_tool_success, format_tool_failure,
    _is_superseded_failure, _is_intermediate_task_result, _verified_visual_response)
from .llm_provider import create_llm_provider
from .model_tool_calls import finish_model_response, speech_filter, looks_like_tool_markup
from core.phrasing import say



def _language_policy() -> str:
    """Hindi/Hinglish understanding and reply-language rules for the prompt."""
    hindi_voice = str(os.getenv("ASTA_TTS_HINDI", "1")).strip().lower() not in {"0", "false", "no", "off"}
    script = (
        "simple conversational Hindi written in Devanagari script (keep app names, "
        "song titles and technical terms in English) so it is spoken with the Hindi voice"
        if hindi_voice
        else "Hinglish written in Latin script (for example 'Haan, main theek hoon')"
    )
    return (
        "LANGUAGE POLICY:\n"
        "The user speaks English, Hindi, or Hinglish (Hindi mixed with English). "
        "Speech transcripts of Hindi arrive in Devanagari or romanised Hindi "
        "(for example 'kya haal hai', 'mujhe ek joke sunao'); understand them as Hindi, "
        "even when a few words are misspelled by the speech recogniser. "
        f"Reply in the user's language: English for English; for Hindi or Hinglish reply in {script}. "
        "Asking to play, open, or search something is a request, not a result: never say you are "
        "playing or opening something unless a tool result confirms it.\n\n"
    )

class AIModule(Module):

    # Bare fillers such as "ok", "okay" and "sure" are deliberately not
    # approvals: they are common conversational noise and must not be able to
    # authorize a high-risk or critical action.
    _APPROVAL_CONFIRMATIONS = {
        "yes", "yeah", "yep", "yup",
        "confirm", "confirmed", "i confirm", "go ahead", "do it",
        "proceed", "yes proceed", "yes do it", "yes go ahead",
    }

    _REJECTED_ERROR = "Tool execution rejected by user."
    _APPROVAL_EXPIRED_MESSAGE = (
        "The approval timed out, so I stopped the task before that step."
    )

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

    # Speech often hears "turn off" as "then of" / "done of" / "turn of".
    _CONVERSATION_MODE_OFF_HEARD = re.compile(
        r"(?:turn|then|than|done|tern|turned|ten|den|switch|shut)\s+(?:off|of)\s+"
        r"(?:the\s+)?conversations?\s+mode"
        r"|(?:turn|switch)\s+(?:the\s+)?conversations?\s+mode\s+(?:off|of)"
        r"|conversations?\s+mode\s+of"
    )

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

        self.engine = create_llm_provider(kernel=kernel, purpose="chat")
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)
        self.chat_history = shared_history(kernel)
        self._chat_epoch = 0
        self._post_response_capture = PostResponseCapture(self.event_bus)

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
        self.event_bus.subscribe("chat_session_changed", self.on_chat_session_changed)
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe(
            "incremental_user_message",
            self.on_incremental_user_message,
        )
        self.event_bus.subscribe("tool_result", self.on_tool_result)
        self.event_bus.subscribe("task_failed", self.on_task_failed)
        self.event_bus.subscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.subscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.subscribe("tool_confirmation_response", self.on_tool_confirmation_answered)
        print("[AI] Ready", flush=True)

    def shutdown(self):
        capture = getattr(self, "_post_response_capture", None)
        if capture is not None:
            capture.close()
        self.event_bus.unsubscribe("chat_session_changed", self.on_chat_session_changed)
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe(
            "incremental_user_message",
            self.on_incremental_user_message,
        )
        self.event_bus.unsubscribe("tool_result", self.on_tool_result)
        self.event_bus.unsubscribe("task_failed", self.on_task_failed)
        self.event_bus.unsubscribe("speech_interrupt", self.on_speech_interrupt)
        self.event_bus.unsubscribe(
            "tool_confirmation_required",
            self.on_tool_confirmation_required,
        )
        self.event_bus.unsubscribe("tool_confirmation_response", self.on_tool_confirmation_answered)
        if not getattr(self.kernel, "_owns_chat_history", False):
            self.chat_history.close()

        engine_shutdown = getattr(self.engine, "shutdown", None)
        if callable(engine_shutdown):
            engine_shutdown()

        decision_shutdown = getattr(self.kernel.decision_engine, "shutdown", None)
        if callable(decision_shutdown):
            decision_shutdown()

        print("[AI] Stopped", flush=True)

    def _registered_tool_names(self):
        try:
            return [definition.name for definition in self.kernel.tool_registry.definitions()]
        except Exception:
            return []

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
            "Mention your creator only when the user explicitly asks who created, made, built, or developed you. "
            "Only if the user's message is a misheard fragment of a few words that forms no request, "
            "say \"Sorry, I didn't catch that. Could you say it again?\" and do not introduce yourself. "
            "For a full sentence, answer it; if it asks for an action that was not carried out, say briefly what you "
            "understood and ask the user to confirm or rephrase, using your most recent action when it is relevant. "
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
            f"{_language_policy()}"
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

    def on_speech_interrupt(self, *args, **kwargs):
        """Stop the streaming answer when the user barges in.

        Without this the LLM kept streaming, and the next sentence it produced
        re-armed the speech worker, so A.S.T.A. resumed talking after "stop".
        """
        self._cancel_post_response_capture()
        cancel = getattr(self.engine, "cancel", None)
        if callable(cancel):
            cancel()

    def on_incremental_user_message(self, text, *args, **kwargs):
        """Process non-command incremental speech without duplicating runtime plans."""
        # Sessionized incremental commands are orchestrated by TaskRuntime so
        # queued continuations cannot bypass its serialization boundary.
        session_id = str(kwargs.get("session_id") or "").strip()
        if session_id:
            intent = self.kernel.intent_router.analyze(text)
            if intent.intent == IntentType.COMMAND:
                print(
                    "[AI] Incremental command already owned by TaskRuntime: "
                    f"{text!r}",
                    flush=True,
                )
                return
        self.on_user_message(text)

    def on_user_message(self, text):
        if not text:
            return
        self._cancel_post_response_capture()
        print(f"[AI] User: {text}", flush=True)
        self._current_user_text = str(text)

        if self._handle_approval_response(text):
            return

        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and background.handle_control(text):
            return
        if background is not None and background.busy:
            hint = self.kernel.intent_router.analyze(text)
            if self._background_media_control(hint):
                return
            if self._side_actions().try_start(text, hint):
                return
            if hint.intent in {IntentType.COMMAND, IntentType.TASK} or re.match(
                r"^(?:(?:please|can you|could you|would you)\s+)*(?:open|launch|start|close|run|stop|play|pause|resume|press|type|switch|fix|write|create|modify|delete)\b",
                str(text).strip(), re.I):
                self._emit_assistant_text("A project task is still running, and that would touch the project too, so it has to wait. I can keep chatting, or open apps and play music meanwhile.")
                return

        if self._handle_post_response_screenshot(text):
            return

        if handle_shortcut(self, text):
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

        quick = quick_math(text)
        if quick:
            print(f"[AI] System 1 quick answer: {quick}", flush=True)
            self._emit_assistant_text(quick)
            return

        file_reply = unsupported_file_action(text, self._registered_tool_names())
        if file_reply:
            print("[AI] No file tool registered; answering honestly", flush=True)
            self._emit_assistant_text(file_reply)
            return

        if self._handle_work_followup(text):
            return

        if work_awareness.is_acknowledgement(text):
            work = work_awareness.recent_work(getattr(self.kernel, "task_manager", None), max_age=180)
            if work is not None:
                # "Okay" right after "Playing Baithi Hai": the LLM used to answer
                # "Okay, I'll play Baithi Hai" -- a promise of an action that never runs.
                print(f"[AI] Acknowledgement after recent work ({work.goal}); staying quiet.", flush=True)
                return

        if self._is_unknown_name_question(text):
            self._emit_assistant_text(
                "I don't know your name yet. I don't have that information stored."
            )
            return

        intent_hint = self.kernel.intent_router.analyze(text)
        if intent_hint.intent in {IntentType.UNKNOWN, IntentType.CONVERSATION}:
            followup = project_followup.followup_intent(text, getattr(self.kernel, "task_manager", None))
            if followup is not None:
                print(f"[AI] Follow-up to the last project task: {followup.normalized_text}", flush=True)
                intent_hint = followup
        if intent_hint.intent in {IntentType.UNKNOWN, IntentType.CONVERSATION}:
            understood = self._understand_project_request(text)
            if understood is True:
                return  # asked what to change
            if understood is not None:
                intent_hint = understood
        recovered_intent = self._recover_recent_command(text, intent_hint)
        if recovered_intent is not None:
            intent_hint = recovered_intent
        contextual = self._resolve_contextual_intent(text, intent_hint)
        if contextual is not None:
            intent_hint = contextual

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
        if result.intent == IntentType.TASK:
            task_runtime = getattr(self.kernel, "task_runtime", None)
            starter = getattr(task_runtime, "start_goal", None)
            if not callable(starter):
                self._emit_assistant_text(
                    "I can't start a project task because the task runtime is unavailable."
                )
                return
            self._start_project_task(text, result)
            return
        if result.intent == IntentType.MEMORY:
            self.event_bus.emit("memory_request", intent=result)
            # Nothing used to answer memory requests, so "remember that..."
            # produced complete silence.
            memory = getattr(self.kernel, "memory", None)
            if bool(getattr(memory, "available", False)):
                self._emit_assistant_text("Got it. I'll remember that.")
            else:
                self._emit_assistant_text(
                    "I heard you, but long-term memory isn't enabled right now, "
                    "so I can't save that permanently."
                )
            return

        capability_response = self._capability_response(text)
        if capability_response is not None:
            self._emit_assistant_text(capability_response)
            return

        work = work_awareness.recent_work(getattr(self.kernel, "task_manager", None))
        note = work_awareness.llm_context(work)
        self._generate_response(text, runtime_context=f"[{note}]\n{text}" if note else None)

    def _start_project_task(self, text, result):
        if result.classifier in {"project_followup", "project_conversation"}:
            text = str((result.entities or {}).get("goal") or text)
        task_runtime = getattr(self.kernel, "task_runtime", None)
        starter = getattr(task_runtime, "start_goal", None)
        background = getattr(self.kernel, "background_tasks", None)
        if background is not None:
            background.start(text, result)
            return
        if not callable(starter):
            self._emit_assistant_text("I can't start a project task because the task runtime is unavailable.")
            return
        task = starter(text, result)
        if task is None:
            self._emit_assistant_text(
                "I couldn't create a safe executable plan for that project task. "
                "No project actions were started."
            )

    def _project_conversation(self):
        talk = getattr(self, "_project_talk", None)
        if talk is None:
            talk = self._project_talk = ProjectConversation()
        return talk

    def _understand_project_request(self, text, *, forced=False):
        """Project work said conversationally: a TASK intent, True (asked a question), or None."""
        manager = getattr(self.kernel, "workspace_manager", None)
        snapshot = manager.snapshot() if manager is not None else {}
        project_path = str(snapshot.get("project_path") or "")
        complete = getattr(self.engine, "complete_json", None)
        if not project_path or not callable(complete) or os.getenv(
            "ASTA_INTENT_LLM", "1"
        ).strip().lower() in {"0", "false", "off", "no"}:
            return None
        try:
            from core.tools.workspace import planning_file_context

            files = list((planning_file_context(manager, "") or {}).get("files") or [])
        except Exception:
            files = []
        last = project_followup.last_project_task(getattr(self.kernel, "task_manager", None))
        talk = self._project_conversation()
        if not forced and not talk.worth_checking(text, files, last):
            return None
        try:
            history = self._active_chat_context(limit=8)
        except Exception:
            history = []
        recent = ""
        if last is not None:
            status = getattr(getattr(last, "status", None), "value", "") or ""
            recent = f"{last.goal} ({status})"
        project = str(snapshot.get("project_name") or "") or re.split(r"[\\/]", project_path.rstrip("\\/"))[-1]
        started = time.perf_counter()
        decision = talk.decide(text, complete, project=project, files=files, recent_task=recent, history=history)
        print(
            f"[AI] Project conversation: {decision.action if decision else 'unparsed'}"
            f" goal={getattr(decision, 'goal', '')!r} question={getattr(decision, 'question', '')!r}"
            f" ({time.perf_counter() - started:.2f}s)",
            flush=True,
        )
        if decision is None or decision.action == "none":
            return None
        if decision.action == "ask":
            self._emit_assistant_text(decision.question)
            return True
        entities = self.kernel.intent_router._extract_project_task(f"edit the file {decision.goal}") or {}
        if not entities:
            return None
        entities["goal"] = decision.goal
        return IntentResult(
            intent=IntentType.TASK,
            confidence=0.8,
            normalized_text=decision.goal,
            entities=entities,
            requires_tools=True,
            requires_reasoning=True,
            classifier="project_conversation",
        )

    def _project_request_from_tool_call(self):
        """The chat model reached for project files: understand the request instead."""
        text = getattr(self, "_current_user_text", "")
        understood = self._understand_project_request(text, forced=True) if text else None
        if understood is None:
            return False
        if understood is not True:
            print(f"[AI] Intent: task (classifier=project_conversation)", flush=True)
            self._start_project_task(text, understood)
        return True

    def _handle_work_followup(self, text):
        """"Try again" / "what happened?" refer to the task that just ran."""
        retry = work_awareness.is_retry_request(text)
        status = not retry and work_awareness.is_status_question(text)
        if not (retry or status):
            return False
        work = work_awareness.recent_work(getattr(self.kernel, "task_manager", None))
        if status:
            if work is None:
                return False  # nothing of ours to explain; let the LLM answer
            print(f"[AI] Explaining recent work: {work.goal} ({work.status})", flush=True)
            self._emit_assistant_text(work_awareness.describe(work))
            return True
        if work is None:
            if self.kernel.intent_router.analyze(text).intent == IntentType.COMMAND:
                return False  # "play it again" with nothing to retry = resume
            self._emit_assistant_text("There's nothing recent for me to retry. What would you like me to do?")
            return True
        if not work.failed and self.kernel.intent_router.analyze(text).intent == IntentType.COMMAND:
            return False  # last task worked; "play it again" keeps its own meaning
        previous = getattr(self, "_retried_goal", None)
        if work.failed and previous == work.goal:
            # Repeating the identical plan a third time won't change anything.
            reason = work_awareness._plain_error(work.error)
            self._emit_assistant_text(
                "I already retried that and it failed the same way"
                + (f": {reason}." if reason else ".")
                + " Could you say it differently, or tell me what to change?"
            )
            self._retried_goal = None
            return True
        self._retried_goal = work.goal
        print(f"[AI] Retrying recent task: {work.goal} (was {work.status})", flush=True)
        self.on_user_message(work.goal)
        return True

    _REFERENCE_WORDS = re.compile(
        r"\b(?:it|this|that|these|those|screen|ise|isko|isse|ye|yeh|wo|woh|usko|use)\b", re.IGNORECASE
    )
    _HINGLISH_VERBS = re.compile(
        r"\b(?:chala\w*|baja\w*|band|khol\w*|bajao|lagao|laga\w*|rok\w*|hata\w*|dikha\w*)\b", re.IGNORECASE
    )

    def _resolve_contextual_intent(self, text, intent):
        """Commands the rule router missed, resolved against recent work.

        "Baithi Hai is on the screen, play it" and other free phrasings go
        through a screen-reference rule, then (for anything else that reads
        like a request) a JSON-only LLM pass that picks one action.
        """
        work = work_awareness.recent_work(getattr(self.kernel, "task_manager", None))
        entities = intent_resolver.resolve_screen_reference(text, work)
        classifier = "screen_reference"
        if entities is None:
            if intent.intent is not IntentType.UNKNOWN or not self._worth_llm_intent(text, work):
                return None
            complete = getattr(self.engine, "complete_json", None)
            if not callable(complete) or os.getenv("ASTA_INTENT_LLM", "1").strip().lower() in {"0", "false", "off", "no"}:
                return None
            applications = getattr(self.kernel, "application_manager", None)
            recent = getattr(applications, "last_opened_application", None) if applications is not None else None
            last_app = str(getattr(recent, "name", recent) or "")
            started = time.perf_counter()
            resolution = intent_resolver.resolve_with_llm(text, complete, work=work, last_app=last_app)
            print(
                f"[AI] LLM intent: {resolution.action if resolution else 'unparsed'}"
                f" query={getattr(resolution, 'query', '')!r} app={getattr(resolution, 'app', '')!r}"
                f" ({time.perf_counter() - started:.2f}s)",
                flush=True,
            )
            if resolution is None or resolution.action == "none":
                return None
            if resolution.action == "retry":
                if work is None:
                    return None
                retried = self.kernel.intent_router.analyze(work.goal)
                return retried if retried.intent is IntentType.COMMAND else None
            manager = getattr(self.kernel, "media_manager", None)
            provider_for = getattr(manager, "provider_for_application", None)
            entities = intent_resolver.resolution_entities(
                resolution, work, provider_for if callable(provider_for) else None
            )
            classifier = "llm_resolver"
        if not entities:
            return None
        if entities.get("on_screen"):
            where = entities.get("provider") or "the app"
            goal = f"play {entities['query']} from the screen in {where}"
        else:
            goal = " ".join(str(text).strip().lower().split())
        print(f"[AI] Understood from context ({classifier}): {entities}", flush=True)
        return IntentResult(
            intent=IntentType.COMMAND,
            confidence=0.85,
            normalized_text=goal,
            entities=entities,
            requires_tools=True,
            classifier=classifier,
        )

    def _worth_llm_intent(self, text, work):
        """Only sentences that read like requests pay for an LLM pass."""
        words = re.findall(r"\w+", str(text))
        if len(words) < 2:
            return False
        if self._looks_like_action_request(text) or self._HINGLISH_VERBS.search(str(text)):
            return True
        return work is not None and bool(self._REFERENCE_WORDS.search(str(text)))

    def _recover_recent_command(
        self,
        text: str,
        intent: IntentResult,
    ) -> IntentResult | None:
        """Recover a clipped/garbled media command from a very recent plan."""
        task_manager = getattr(self.kernel, "task_manager", None)
        if task_manager is None:
            return None

        # Only use very recent task context. This is context recovery, not a
        # general command override.
        tasks = task_manager.list()
        if not tasks:
            return None

        recent = max(
            tasks,
            key=lambda task: getattr(
                task,
                "updated_at",
                datetime.min.replace(tzinfo=timezone.utc),
            ),
        )
        updated_at = getattr(recent, "updated_at", None)
        if updated_at is None:
            return None

        try:
            age_seconds = (
                datetime.now(timezone.utc) - updated_at
            ).total_seconds()
        except TypeError:
            return None

        if age_seconds < 0 or age_seconds > 20:
            return None

        plan = getattr(recent, "plan", None)
        if plan is None:
            return None

        current_query = ""
        current_media = (
            intent.intent is IntentType.COMMAND
            and intent.entities.get("action") == "media"
            and str(intent.entities.get("operation") or "").lower() == "play"
        )
        if current_media:
            current_query = str(intent.entities.get("query") or "").strip()

        transcript_tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        if not transcript_tokens:
            return None

        for step in plan.steps:
            metadata = step.metadata or {}
            if str(metadata.get("action") or "").strip().lower() != "media":
                continue

            query = str(metadata.get("query") or "").strip()
            operation = str(metadata.get("operation") or "").strip().lower()
            if operation != "play" or not query:
                continue

            query_tokens = re.findall(r"[a-z0-9]+", query.lower())
            if not query_tokens:
                continue

            comparison_text = current_query or str(text)
            comparison_tokens = re.findall(r"[a-z0-9]+", comparison_text.lower())
            if not comparison_tokens:
                continue

            coverage = []
            for query_token in query_tokens:
                best = max(
                    (
                        SequenceMatcher(
                            None,
                            query_token,
                            candidate,
                            autojunk=False,
                        ).ratio()
                        for candidate in comparison_tokens
                    ),
                    default=0.0,
                )
                coverage.append(best)

            match_score = sum(coverage) / len(coverage)

            # Unknown/STT-fragment input can recover from a reasonably strong
            # match. A recognized media command is only replaced when it looks
            # like a noisy continuation of the same request.
            if intent.intent is IntentType.UNKNOWN:
                acceptable = match_score >= 0.68
            else:
                noisy_tokens = {"and", "the", "a", "an", "sir", "please", "okay", "ok"}
                acceptable = (
                    current_media
                    and match_score >= 0.38
                    and any(token in noisy_tokens for token in comparison_tokens)
                    and current_query != query
                )

            if not acceptable:
                continue

            entities = {
                "action": "media",
                "operation": operation,
                "query": query,
            }
            provider = str(metadata.get("provider") or "").strip()
            if provider:
                entities["provider"] = provider

            print(
                "[AI] Recovered media command from recent task context: "
                f"{operation} {query} (match={match_score:.2f})",
                flush=True,
            )
            return IntentResult(
                intent=IntentType.COMMAND,
                confidence=max(0.80, min(0.98, match_score)),
                normalized_text=" ".join(
                    str(text).strip().lower().split()
                ),
                entities=entities,
                requires_tools=True,
                classifier="task_context",
            )

        return None

    def _run_system1_decision(self, text, *, intent_hint=None):
        if (
            intent_hint is not None
            and intent_hint.intent is IntentType.TASK
        ):
            # Scoped project goals use the task planner, not the single-action
            # System-1 command resolver.
            return False

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
                    else:
                        # No rule target (unrecognised phrasing such as
                        # "fire up chrome for me"): offer apps that match
                        # individual words so Laya can pick the target.
                        candidates.extend(
                            self._candidate_applications(manager, text)
                        )

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
                mentioned = str(
                    action_decision.arguments.get("target_app") or ""
                ).strip()
                manager = getattr(self.kernel, "application_manager", None)
                if mentioned and manager is not None:
                    manager.last_mentioned_application = mentioned

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

                target_app = str(
                    action_decision.arguments.get("target_app") or ""
                ).strip()
                if (
                    intent_hint.intent != IntentType.COMMAND
                    and action_decision.is_actionable
                    and action_decision.action
                    in (ActionType.OPEN_APP, ActionType.CLOSE_APP)
                    and target_app
                    and action_decision.command_complete
                    and not action_decision.compound
                    and action_decision.confidence >= 0.80
                ):
                    app_intent = IntentResult(
                        intent=IntentType.COMMAND,
                        confidence=action_decision.confidence,
                        normalized_text=" ".join(
                            str(text).strip().lower().split()
                        ),
                        entities={
                            "action": (
                                "open"
                                if action_decision.action is ActionType.OPEN_APP
                                else "close"
                            ),
                            "target": target_app,
                        },
                        requires_tools=True,
                        classifier="laya_system1",
                    )
                    print(
                        f"[AI] System 1 recovered an application command: "
                        f"{app_intent.entities['action']} {target_app}",
                        flush=True,
                    )
                    self._handle_command_intent(app_intent)
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

    _CANDIDATE_STOPWORDS = frozenset(
        """
        a an the to for me my please can could would will you it this that up
        open launch start close run stop quit exit fire bring show switch go
        and then now app application program hey okay ok just
        """.split()
    )

    @classmethod
    def _candidate_applications(cls, manager, text, *, limit=12):
        """Installed apps that closely match individual words of ``text``."""
        found = []
        seen = set()
        for word in re.findall(r"[a-z0-9][a-z0-9.+#-]*", str(text).lower()):
            if len(word) < 3 or word in cls._CANDIDATE_STOPWORDS:
                continue
            try:
                matches = manager.discover(word, limit=3)
            except Exception:
                continue
            for app in matches:
                name = str(getattr(app, "name", app) or "").strip()
                if not name or name.lower() in seen:
                    continue
                seen.add(name.lower())
                found.append(app)
                if len(found) >= limit:
                    return found
        return found

    @staticmethod
    def _looks_like_action_request(text):
        normalized = " ".join(str(text).strip().lower().split())
        return bool(
            re.search(
                r"\b(?:open|launch|start|close|run|stop|play|pause|resume|skip|next|previous|"
                r"back|screenshot|capture|mute|unmute|scroll|press|type|fire|bring|switch|quit|exit|"
                r"kill|load)\b",
                normalized,
            )
        )

    def _active_chat_context(self, *, limit):
        store = self.chat_history
        read = getattr(store, "messages_for_session", None)
        if callable(read):
            return read(store.session_id, limit=limit)
        return store.latest_active_context(limit=limit)  # legacy test adapters

    def on_chat_session_changed(self, session_id):
        self._chat_epoch = getattr(self, "_chat_epoch", 0) + 1
        self._cancel_post_response_capture()
        cancel = getattr(self.engine, "cancel", None)
        if callable(cancel):
            cancel()
        reset = getattr(self.engine, "reset_conversation", None)
        if callable(reset):
            reset()

    def _context_response(self, text):
        normalized = self._normalize_question(text)
        if normalized not in self._CONTEXT_QUESTIONS:
            return None

        messages = self._active_chat_context(limit=12)
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

        if (
            normalized in self._CONVERSATION_MODE_OFF
            or self._CONVERSATION_MODE_OFF_HEARD.fullmatch(normalized)
        ):
            self.event_bus.emit("conversation_mode_set", enabled=False)
            self._emit_assistant_text(
                "Conversation mode is off. Say the wake word when you need me."
            )
            return True

        return False

    @classmethod
    def _classify_short_approval(cls, compact):
        """Use the same complete-utterance consent policy as voice input."""
        return classify_approval(compact)

    def _handle_approval_response(self, text):
        manager = self.kernel.approval_manager

        # Expired approvals are rejected so their task fails cleanly and a
        # later "yes" can never execute them.
        list_expired = getattr(manager, "list_expired", None)
        if callable(list_expired):
            for stale in list_expired():
                print(
                    "[AI] Approval expired "
                    f"(request_id={stale.request.request_id})",
                    flush=True,
                )
                self.event_bus.emit(
                    "tool_confirmation_response",
                    request_id=stale.request.request_id,
                    approved=False,
                )

        pending = manager.list_pending()
        if not pending:
            return False

        background = getattr(self.kernel, "background_tasks", None)
        background_pending = [item for item in pending
            if background is not None and background.owns(item.request.metadata.get("task_id"))]
        if background_pending:
            compact = normalize_confirmation_transcript(text)
            explicit = {"approve coding step": True, "approve project step": True,
                        "reject coding step": False, "reject project step": False}
            if compact in explicit and len(pending) == 1:
                self.event_bus.emit("tool_confirmation_response",
                    request_id=pending[0].request.request_id, approved=explicit[compact])
                return True
            if compact in self._APPROVAL_CONFIRMATIONS or compact in self._APPROVAL_REJECTIONS:
                self._emit_assistant_text("For the background task, use the approval button or say approve coding step or reject coding step.")
                return True
            # Other chat does not silently approve or cancel background work.
            return False

        approved, rejected = classify_approval(text)

        if not (approved or rejected):
            # The user moved on to something else. Cancel the pending request
            # instead of leaving it armed for an unrelated future "yes".
            for item in pending:
                print(
                    "[AI] Approval superseded by a new request "
                    f"(request_id={item.request.request_id})",
                    flush=True,
                )
                self.event_bus.emit(
                    "tool_confirmation_response",
                    request_id=item.request.request_id,
                    approved=False,
                )
            return False

        # A yes/no applies only to the most recent request. Older requests are
        # cancelled so an answer cannot silently authorize the wrong action.
        ordered = sorted(pending, key=lambda item: item.created_at)
        for older in ordered[:-1]:
            self.event_bus.emit(
                "tool_confirmation_response",
                request_id=older.request.request_id,
                approved=False,
            )

        request = ordered[-1].request
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
        return normalized in variants or is_creator_question(text)

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
            "For my current version, the AI runs locally through llama.cpp, speech recognition uses Whisper, and speech synthesis uses Kokoro, so the core interaction can run locally on the machine.",
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

    # Only questions about executable abilities get the capability list.
    # Previously any "can you ..." / "do you have ..." question matched, so
    # "can you tell me a joke" answered with a list of tool names.
    _CAPABILITY_QUESTION_PATTERN = re.compile(
        r"^(?:"
        r"what (?:can you do|are your capabilities|tools do you have|actions can you (?:do|perform|take))"
        r"|what are you capable of"
        r"|(?:which|what) (?:tools|capabilities|integrations) do you (?:have|support)"
        r"|do you (?:have|support) (?:any )?(?:tools|capabilities|integrations)"
        r"|(?:can|could) (?:you|asta) (?:control|automate) (?:my )?(?:computer|pc|apps|applications|system)"
        r"|list (?:your )?(?:tools|capabilities)"
        r")\b"
    )

    def _capability_response(self, text):
        normalized = self._normalize_question(text)
        if not self._CAPABILITY_QUESTION_PATTERN.match(normalized):
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

    def on_task_failed(self, task=None, *args, **kwargs):
        """Give the user an explicit terminal response when autonomous work fails."""
        if isinstance(task, dict):
            goal = str(task.get("goal") or "").strip()
            error = str(task.get("error") or "")
        else:
            goal = str(getattr(task, "goal", "") or "").strip()
            error = str(getattr(task, "error", "") or "")

        # A user rejection (or a superseded/expired approval) is already
        # acknowledged by the rejected tool result; do not also announce it as
        # a task failure.
        if error == self._REJECTED_ERROR:
            return
        if error.startswith("The approval expired"):
            # Same text as the tool-result message, so it is said only once.
            self._emit_assistant_text(self._APPROVAL_EXPIRED_MESSAGE)
            return

        # The failed step explains why; don't read the whole request back.
        message = "Sorry, I couldn't finish that."
        from core.tools.sandbox import SANDBOX_SETUP_MESSAGE

        metadata = (task.get("metadata") if isinstance(task, dict) else getattr(task, "metadata", None)) or {}
        setup = metadata.get("capability_setup") if isinstance(metadata, dict) else None
        if error == SANDBOX_SETUP_MESSAGE or (
            isinstance(setup, dict) and setup.get("capability") == "sandbox" and error
        ):
            # Sandbox repair results say what was checked and what to do.
            message = error
        print(f"[AI] Task failed: {goal or message}", flush=True)
        self._emit_assistant_text(message)

    _SIDE_MEDIA_OPERATIONS = frozenset({"pause", "play", "toggle", "next", "previous", "stop"})

    def _side_actions(self):
        """Open apps / play music now, next to a background project task."""
        runner = getattr(self, "_side_runner", None)
        if runner is None:
            from core.side_actions import SideActionRunner

            runner = self._side_runner = SideActionRunner(self.kernel, say=self._emit_assistant_text)
        return runner

    def _background_media_control(self, intent) -> bool:
        """Pause/resume/skip music while a project task runs in the background.

        These don't touch the project, so they need not wait for it. Only
        bare transport controls qualify; "play <song>" still waits.
        """
        entities = getattr(intent, "entities", None) or {}
        if getattr(intent, "intent", None) is not IntentType.COMMAND:
            return False
        if str(entities.get("action") or "").lower() != "media" or entities.get("commands"):
            return False
        operation = str(entities.get("operation") or "").lower()
        if operation not in self._SIDE_MEDIA_OPERATIONS or entities.get("query"):
            return False
        import uuid

        request = ToolRequest(
            tool="media.control",
            arguments={"operation": operation},
            request_id=f"side-media-{uuid.uuid4().hex[:12]}",
            metadata={"action": "media", "side_action": True},
        )
        result = self.kernel.tool_dispatcher.dispatch(request)
        print(
            f"[AI] Media control during project task: {operation} -> "
            f"{'ok' if result.success else result.error}",
            flush=True,
        )
        if result.success:
            self._emit_assistant_text(self._format_tool_success(result))
        else:
            self._emit_assistant_text("I couldn't control media playback.")
        return True

    def on_tool_confirmation_required(self, request, reason):
        action = request.metadata.get("action")
        target = request.arguments.get("target")
        overwrite = str(request.metadata.get("confirm_overwrite") or "").strip()
        if overwrite and request.metadata.get("confirm_kind") == "test_edit":
            text = (
                f"To finish, I'd have to change the test file {overwrite}, which you didn't ask me to edit. "
                "Should I change it?"
            )
        elif overwrite:
            text = f"{overwrite} already exists with different content. Should I replace it?"
        elif request.tool == "project.open_program":
            text = (
                f"Can I open {request.arguments.get('path') or 'the program'} on your desktop? "
                "It runs outside the sandbox, with your normal user permissions."
            )
        elif action and target:
            text = f"I need your confirmation before I {action} {target}."
        elif target:
            text = f"I need your confirmation before acting on {target}."
        else:
            text = "I need your confirmation before performing that action."
        background = getattr(self.kernel, "background_tasks", None)
        in_background = background is not None and background.owns(request.metadata.get("task_id"))
        if in_background:
            text += " Use the approval button, or say approve coding step or reject coding step."
        print(f"[AI] Approval required: {reason}", flush=True)
        spoken = self._spoken_approval(request, in_background) or None
        questions = getattr(self, "_approval_questions", None)
        if questions is None:
            questions = self._approval_questions = {}
        questions[str(request.request_id)] = spoken or text
        while len(questions) > 8:
            questions.pop(next(iter(questions)))
        self._emit_assistant_text(text, spoken=spoken)

    def on_tool_confirmation_answered(self, request_id=None, approved=None, **_kwargs):
        """Approved/rejected before the question was read out: don't ask it anymore.

        Clicking Approve while A.S.T.A. was still speaking the previous line
        used to leave the old question queued, so it was asked right before
        "Done, it's open".
        """
        question = (getattr(self, "_approval_questions", None) or {}).pop(str(request_id), None)
        if question:
            self.event_bus.emit("speech_retract", text=question)

    @staticmethod
    def _spoken_approval(request, in_background) -> str:
        """One short question; the chat card carries the details."""
        overwrite = str(request.metadata.get("confirm_overwrite") or "").strip()
        if overwrite and request.metadata.get("confirm_kind") == "test_edit":
            question = say("approve_test_edit", path=overwrite)
        elif overwrite:
            question = say("approve_replace", path=overwrite)
        elif request.tool == "project.open_program":
            question = say("approve_open", path=str(request.arguments.get("path") or "the program"))
        elif request.tool == "system.run_command":
            target = str(request.arguments.get("target") or "").replace("\\", "/").rsplit("/", 1)[-1]
            target = re.sub(r"\.exe$", "", target, flags=re.IGNORECASE)
            if re.fullmatch(r"python[\d.]*w?", target, flags=re.IGNORECASE):
                target = "python"  # say "python", not "python.exe" or "python3.13"
            arguments = [str(item) for item in (request.arguments.get("arguments") or [])][:3]
            command = " ".join([target, *arguments]).strip() or "the command"
            question = say("approve_run", command=command)
        else:
            return ""  # keep the full sentence
        return question + (say("approve_suffix") if in_background else "")

    def on_tool_result(self, result):
        if not isinstance(result, ToolResult):
            return
        if _is_superseded_failure(self, result) or _is_intermediate_task_result(self, result):
            return
        verified_response = _verified_visual_response(result)
        if verified_response is not None:
            self._emit_assistant_text(verified_response)
            return
        result_metadata = (
            result.metadata if isinstance(result.metadata, dict) else {}
        )
        task_id = result_metadata.get("task_id")
        task = (
            self.kernel.task_manager.get(task_id)
            if isinstance(task_id, str) and task_id
            else None
        )
        if (
            task is not None
            and (getattr(task, "metadata", {}) or {}).get("intent_type")
            == IntentType.TASK.value
        ):
            # TaskRuntime owns the verified user-facing summary for project
            # work. Do not race it with raw intermediate tool output.
            return
        text = (
            self._format_tool_success(result)
            if result.success
            else self._format_tool_failure(result)
        )
        self._emit_assistant_text(text)

    def _emit_assistant_text(self, text, spoken=None):
        if not text:
            return
        # A retried step can fail twice in a row; say the same thing once.
        now = time.monotonic()
        last = getattr(self, "_last_assistant_emit", None)
        if last is not None and last[0] == text and now - last[1] < 5.0:
            print(f"[AI] Suppressed repeated message: {text}", flush=True)
            return
        self._last_assistant_emit = (text, now)
        if spoken is not None and spoken != text:
            self.event_bus.emit("assistant_sentence", text=text, spoken=spoken)
        else:
            self.event_bus.emit("assistant_sentence", text=text)
        self.event_bus.emit("assistant_response", text=text)

    @staticmethod
    def _format_tool_success(result: ToolResult) -> str:
        return format_tool_success(result)

    @staticmethod
    def _format_tool_failure(result: ToolResult) -> str:
        return format_tool_failure(result, rejected_error=AIModule._REJECTED_ERROR,
                                   approval_expired_message=AIModule._APPROVAL_EXPIRED_MESSAGE)

    def _cancel_post_response_capture(self):
        capture = getattr(self, "_post_response_capture", None)
        if capture is not None:
            capture.cancel()

    def _handle_post_response_screenshot(self, text):
        prompt = extract_prompt(text)
        if not prompt:
            return False
        routed = self.kernel.intent_router.analyze(text)
        commands = (routed.entities or {}).get("commands")
        if routed.intent == IntentType.COMMAND and isinstance(commands, list) and len(commands) >= 2:
            return False
        intent = IntentResult(
            intent=IntentType.COMMAND, confidence=1.0,
            normalized_text="take a screenshot", entities={"action": "screenshot"},
            requires_tools=True, classifier="post_response_capture",
        )
        try:
            request = self.tool_request_builder.build(intent)
        except ValueError as exc:
            self._emit_assistant_text(f"I couldn't take the screenshot: {exc}")
            return True
        request.metadata["post_response_action"] = True
        capture = getattr(self, "_post_response_capture", None)
        if capture is None:
            capture = self._post_response_capture = PostResponseCapture(self.event_bus)
        token = capture.begin(request)
        if token is None:
            return True
        try:
            response = self._generate_response(prompt)
        except BaseException:
            capture.cancel(token)
            raise
        capture.complete(token, response)
        return True

    def _prepare_runtime_context(self, text, runtime_context=None):
        """Apply the existing per-turn context policy without a startup wrapper."""
        builder = getattr(self.kernel, "context_builder", None)
        if builder is None:
            builder = ContextBuilder(self.kernel)
            self.kernel.context_builder = builder

        context = runtime_context
        try:
            if not context:
                intent = self.kernel.intent_router.analyze(text)
                context = builder.build_prompt(text, intent)
        except Exception as exc:  # context enrichment must not break chat
            print(
                f"[Context] Failed to build runtime context: {type(exc).__name__}: {exc}",
                flush=True,
            )
            context = None
        return context

    def _generate_response(self, text, runtime_context=None):
        session_id = getattr(self.kernel, "session_id", None)
        epoch = getattr(self, "_chat_epoch", 0)
        def still_current():
            return (getattr(self.kernel, "session_id", None) == session_id
                    and getattr(self, "_chat_epoch", 0) == epoch)
        context = self._prepare_runtime_context(text, runtime_context)
        def on_sentence(sentence):
            if sentence and still_current():
                self.event_bus.emit("assistant_sentence", text=sentence)

        response = self.engine.generate_response(
            text,
            on_sentence=speech_filter(on_sentence),
            context=context,
        )
        if not still_current():
            return None
        if not response:
            print("[AI] No response generated.", flush=True)
            return
        # Tool-call markup is routed (allowlisted) or dropped, never shown.
        contained_tool_markup = looks_like_tool_markup(response)
        response = finish_model_response(self, response)
        if not response:
            return
        self.event_bus.emit("assistant_response", text=response)
        if not contained_tool_markup:
            self.event_bus.emit("conversation_turn_completed", text=response, session_id=session_id)
        return None if contained_tool_markup else response
