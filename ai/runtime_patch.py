import re
import time

from core.contracts import IntentResult, IntentType, ToolResult


_POST_RESPONSE_SCREENSHOT_PATTERN = re.compile(
    r"^(?P<prompt>.+?)\s*(?:,?\s+)(?:and\s+then|then|after that|followed by|and)\s+"
    r"(?:please\s+)?"
    r"(?P<action>"
    r"(?:take(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot))"
    r"|(?:capture(?:\s+(?:a|the))?\s+(?:a\s+)?(?:screenshot|screen\s*shot))"
    r"|(?:screenshot|screen\s*shot)"
    r")$",
    re.IGNORECASE,
)

_SCREENSHOT_OPEN_PHRASES = {
    "open screenshot",
    "open the screenshot",
    "open latest screenshot",
    "open the latest screenshot",
    "open screen shot",
    "open the screen shot",
    "show screenshot",
    "show the screenshot",
    "show latest screenshot",
    "show the latest screenshot",
    "show screen shot",
    "show the screen shot",
}

_SCREENSHOT_CAPTURE_PATTERN = re.compile(
    r"\b(?:take|capture|grab|snap|make)\b.*\b(?:screenshot|screen\s*shot|screen\s*capture)\b",
    re.IGNORECASE,
)

_CONVERSATION_LEAD_PATTERN = re.compile(
    r"^(?:okay|ok|please)\s*[,;:.!?-]*\s+",
    re.IGNORECASE,
)


def _extract_post_response_screenshot(text):
    if not isinstance(text, str):
        return None

    normalized = " ".join(text.strip().split())
    match = _POST_RESPONSE_SCREENSHOT_PATTERN.match(normalized.rstrip(" .!?;:"))
    if not match:
        return None

    prompt = match.group("prompt").strip(" ,.!?;:")
    if not prompt:
        return None
    return prompt


def _is_screenshot_open_request(text):
    normalized = " ".join(str(text).strip().lower().split()).rstrip(" .!?;:")
    return normalized in _SCREENSHOT_OPEN_PHRASES


def _is_screenshot_capture_request(text):
    """Recognize natural spoken screenshot requests before generic command routing."""
    normalized = " ".join(str(text).strip().lower().split()).rstrip(" .!?;:")
    if not normalized:
        return False

    # A natural-language request that asks for a response and then a screenshot
    # must go through the response-first path below, not the deterministic
    # screenshot-only shortcut.
    if _extract_post_response_screenshot(normalized):
        return False

    # Do not turn negative requests such as "don't take a screenshot" into actions.
    if re.search(r"\b(?:don't|do not|dont|never)\b", normalized):
        return False

    return bool(_SCREENSHOT_CAPTURE_PATTERN.search(normalized))


def _is_creator_identity_question(text):
    normalized = " ".join(str(text).strip().lower().split()).rstrip(" .!?;:")
    normalized = normalized.replace("a.s.t.a", "asta")

    direct = {
        "who is your creator",
        "who is asta's creator",
        "who is asta creator",
        "who is behind you",
        "who is behind asta",
        "who is your developer",
        "who developed asta",
        "who built asta",
        "who made asta",
    }
    if normalized in direct:
        return True

    return bool(
        re.search(
            r"\bwho\b.*\b(?:created|made|built|developed|programmed|designed)\b.*\b(?:you|asta)\b",
            normalized,
        )
    )


def _task_for_tool_result(ai, result):
    task_manager = getattr(ai.kernel, "task_manager", None)
    if task_manager is None or not isinstance(result, ToolResult):
        return None

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    task_id = metadata.get("task_id")
    try:
        task = (
            task_manager.get(task_id)
            if task_id
            else task_manager.current()
        )
    except Exception:
        return None
    return task


def _is_intermediate_task_result(ai, result) -> bool:
    task = _task_for_tool_result(ai, result)
    if task is None:
        return False

    plan = getattr(task, "plan", None)
    plan_step_id = (
        result.metadata.get("plan_step_id")
        if isinstance(result.metadata, dict)
        else None
    )
    steps = getattr(plan, "steps", None) or []
    if plan is None or not plan_step_id or not steps:
        return False

    for index, step in enumerate(steps):
        if getattr(step, "id", None) != plan_step_id:
            continue
        # A result from any step with a later step is an internal progress
        # update. Do not let synchronous nested EventBus dispatch turn it into
        # user-facing speech before the final step has produced its answer.
        return index < (len(steps) - 1)

    return False


def _verified_visual_response(result: ToolResult) -> str | None:
    if not result.success or result.tool != "vision.inspect":
        return None

    output = result.output if isinstance(result.output, dict) else {}
    if not bool(output.get("verified")):
        return None

    summary = " ".join(str(output.get("summary") or "").strip().split())
    if summary:
        return f"Done — {summary.rstrip(' .!?;:')}."

    return "Done — the requested visual condition is confirmed."


def apply_ai_runtime_patch():
    """Add deterministic mixed-request, voice-control, and screenshot behavior."""
    from ai.ai_module import AIModule
    from voice.voice_module import VoiceModule

    if getattr(AIModule, "_asta_runtime_patch_applied", False):
        return

    original_on_user_message = AIModule.on_user_message
    original_on_tool_result = AIModule.on_tool_result
    original_creator_handler = AIModule._is_creator_identity_question
    original_format_tool_success = AIModule._format_tool_success
    original_voice_on_conversation_mode_set = VoiceModule.on_conversation_mode_set
    original_voice_can_listen = VoiceModule._can_listen

    def patched_on_tool_result(self, result):
        if _is_intermediate_task_result(self, result):
            print(
                f"[AI] Suppressing intermediate task result: {result.tool}",
                flush=True,
            )
            return

        final_response = _verified_visual_response(result)
        if final_response is not None:
            print(f"[AI] Final visual verification response: {final_response}", flush=True)
            self._emit_assistant_text(final_response)
            return

        original_on_tool_result(self, result)

    def patched_on_user_message(self, text):
        if isinstance(text, str) and text.strip():
            # Whisper often inserts punctuation after conversational fillers,
            # e.g. "Okay, turn off conversation mode.". Canonicalize only for
            # deterministic conversation-mode handling and leave normal text alone.
            normalized_text = " ".join(text.strip().split())
            conversation_text = _CONVERSATION_LEAD_PATTERN.sub(
                "",
                normalized_text,
                count=1,
            )
            if conversation_text != normalized_text:
                if self._handle_conversation_mode_command(conversation_text):
                    return

            if _is_screenshot_open_request(text):
                screenshot_intent = IntentResult(
                    intent=IntentType.COMMAND,
                    confidence=1.0,
                    normalized_text="open latest screenshot",
                    entities={"action": "open_screenshot"},
                    requires_tools=True,
                    classifier="runtime_patch",
                )
                try:
                    request = self.tool_request_builder.build(screenshot_intent)
                except ValueError as exc:
                    print(
                        f"[AI] Unable to build screenshot-open request: {exc}",
                        flush=True,
                    )
                    self._emit_assistant_text(
                        f"I couldn't open the screenshot: {exc}"
                    )
                    return

                request.metadata["screenshot_open_request"] = True
                print(
                    f"[AI] Selected screenshot-open tool: {request.tool} "
                    f"(request_id={request.request_id})",
                    flush=True,
                )
                self.event_bus.emit("tool_request", request=request)
                return

            # Let the canonical intent router claim compound commands first.
            # This prevents a final screenshot action from stealing a sequence
            # such as "open Chrome, then open camera, then take a screenshot".
            routed = self.kernel.intent_router.analyze(text)
            if routed.intent == IntentType.COMMAND:
                commands = routed.entities.get("commands")
                if isinstance(commands, list) and len(commands) >= 2:
                    original_on_user_message(self, text)
                    return

            if _is_screenshot_capture_request(text):
                screenshot_intent = IntentResult(
                    intent=IntentType.COMMAND,
                    confidence=1.0,
                    normalized_text="take a screenshot",
                    entities={"action": "screenshot"},
                    requires_tools=True,
                    classifier="runtime_patch",
                )
                try:
                    request = self.tool_request_builder.build(screenshot_intent)
                except ValueError as exc:
                    print(
                        f"[AI] Unable to build screenshot request: {exc}",
                        flush=True,
                    )
                    self._emit_assistant_text(
                        f"I couldn't take the screenshot: {exc}"
                    )
                    return

                request.metadata["direct_screenshot_request"] = True
                print(
                    f"[AI] Selected screenshot tool: {request.tool} "
                    f"(request_id={request.request_id})",
                    flush=True,
                )
                self.event_bus.emit("tool_request", request=request)
                return

            if routed.intent != IntentType.COMMAND:
                prompt = _extract_post_response_screenshot(text)
                if prompt:
                    print(
                        "[AI] Mixed request detected: response followed by screenshot.",
                        flush=True,
                    )
                    print(f"[AI] Response prompt: {prompt}", flush=True)

                    self._generate_response(prompt)

                    screenshot_intent = IntentResult(
                        intent=IntentType.COMMAND,
                        confidence=1.0,
                        normalized_text="take a screenshot",
                        entities={"action": "screenshot"},
                        requires_tools=True,
                        classifier="runtime_patch",
                    )
                    try:
                        request = self.tool_request_builder.build(screenshot_intent)
                    except ValueError as exc:
                        print(
                            f"[AI] Unable to build post-response screenshot request: {exc}",
                            flush=True,
                        )
                        self._emit_assistant_text(
                            f"I couldn't take the screenshot: {exc}"
                        )
                        return

                    request.metadata["post_response_action"] = True
                    print(
                        f"[AI] Selected post-response tool: {request.tool} "
                        f"(request_id={request.request_id})",
                        flush=True,
                    )
                    self.event_bus.emit("tool_request", request=request)
                    return

        original_on_user_message(self, text)

    # Plain function wrapped exactly once below. Stacking @classmethod inside
    # classmethod() relied on classmethod descriptor chaining, which Python
    # 3.13 removed; the result was "'classmethod' object is not callable" on
    # every user message.
    def patched_creator_handler(cls, text):
        if original_creator_handler(text):
            return True
        return _is_creator_identity_question(text)

    @staticmethod
    def patched_format_tool_success(result: ToolResult):
        if result.tool == "vision.open_screenshot" and result.success:
            output = result.output
            if isinstance(output, dict) and output.get("path"):
                print(
                    f"[AI] Screenshot opened: {output['path']}",
                    flush=True,
                )
            return "Opened the latest screenshot."
        return original_format_tool_success(result)

    def patched_generate_response(self, text, runtime_context=None):
        """Stream complete sentence chunks to speech while retaining turn context."""
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

        # Speech has already received sentence chunks during generation.
        # Emit the completed response separately for chat/HUD consumers.
        self.event_bus.emit("assistant_response", text=response)

    def patched_voice_on_conversation_mode_set(self, enabled):
        original_voice_on_conversation_mode_set(self, enabled)
        if not enabled:
            # Give the wakeword detector a short settling period after ASTA
            # finishes speaking the "conversation mode is off" response.
            self._asta_wakeword_cooldown_until = time.monotonic() + 2.0

    def patched_voice_can_listen(self):
        if not original_voice_can_listen(self):
            return False

        cooldown_until = getattr(self, "_asta_wakeword_cooldown_until", 0.0)
        if not self._conversation_active and time.monotonic() < cooldown_until:
            return False

        return True

    AIModule.on_tool_result = patched_on_tool_result
    AIModule.on_user_message = patched_on_user_message
    AIModule._is_creator_identity_question = classmethod(patched_creator_handler)
    AIModule._format_tool_success = staticmethod(patched_format_tool_success)
    AIModule._generate_response = patched_generate_response
    VoiceModule.on_conversation_mode_set = patched_voice_on_conversation_mode_set
    VoiceModule._can_listen = patched_voice_can_listen
    AIModule._asta_runtime_patch_applied = True
