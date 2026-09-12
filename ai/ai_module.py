from core.module import Module
from core.contracts import IntentType, IntentResult, ToolResult
from core.tools import ToolRequestBuilder

from .openai_engine import AIEngine


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

    def __init__(self, kernel):
        super().__init__(
            name="AIModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = AIEngine()
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)

    def initialize(self):
        print("[AI] Initializing...", flush=True)
        self._ground_engine_in_capabilities()

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
            "You are ASTA, a local AI voice assistant. "
            "Respond naturally and concisely. "
            "Prefer 1–3 short sentences for normal questions. "
            "Sound conversational, helpful, and direct.\n\n"
            "TRUTHFUL CAPABILITY POLICY:\n"
            "The registered capabilities below are the authoritative list of "
            "actions ASTA can currently execute. Never claim ASTA can perform "
            "an action that is not represented by a registered capability. "
            "Do not say an action was completed unless a tool result confirms "
            "success. If a requested capability is unavailable, say so plainly. "
            "Never invent tool names, integrations, application support, memory, "
            "personal facts, or completed actions.\n\n"
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

        if self._is_unknown_name_question(text):
            self._emit_assistant_text(
                "I don't know your name yet. I don't have that information stored."
            )
            return

        result: IntentResult = self.kernel.intent_router.analyze(text)
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

    def _handle_conversation_mode_command(self, text):
        normalized = self._normalize_question(text)

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

    def _handle_command_sequence(self, intent: IntentResult, commands):
        """Build the first request and carry the remaining sequence in metadata."""
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

        print(
            f"[AI] Compound command: {len(commands)} step(s)",
            flush=True,
        )
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
            if result.tool == "system.stop_process" and output.get("pid"):
                return f"Stopped process {output['pid']}."
            if result.tool == "system.start_process" and target:
                return f"Started {target}."
            if result.tool == "vision.screenshot":
                path = output.get("path")
                if path:
                    print(f"[AI] Screenshot saved: {path}", flush=True)
                return "Screenshot captured."
        if output is None:
            return f"{result.tool} completed successfully."
        return str(output)

    @staticmethod
    def _format_tool_failure(result: ToolResult) -> str:
        if result.error:
            return f"I couldn't complete that action: {result.error}"
        return "I couldn't complete that action."

    def _generate_response(self, text):
        def on_sentence(sentence):
            if sentence:
                self.event_bus.emit("assistant_sentence", text=sentence)

        response = self.engine.generate_response(text, on_sentence=on_sentence)
        if not response:
            print("[AI] No response generated.", flush=True)
            return
        self.event_bus.emit("assistant_response", text=response)
