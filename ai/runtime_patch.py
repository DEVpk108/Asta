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
    "show screenshot",
    "show the screenshot",
    "show latest screenshot",
    "show the latest screenshot",
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


def apply_ai_runtime_patch():
    """Add deterministic mixed-request, voice-control, and screenshot behavior."""
    from ai.ai_module import AIModule
    from voice.voice_module import VoiceModule

    if getattr(AIModule, "_asta_runtime_patch_applied", False):
        return

    original_on_user_message = AIModule.on_user_message
    original_creator_handler = AIModule._is_creator_identity_question
    original_format_tool_success = AIModule._format_tool_success
    original_generate_response = AIModule._generate_response
    original_voice_on_conversation_mode_set = VoiceModule.on_conversation_mode_set
    original_voice_can_listen = VoiceModule._can_listen

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

            routed = self.kernel.intent_router.analyze(text)
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

    @classmethod
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

    def patched_generate_response(self, text):
        """Emit a completed response once so abbreviations do not become TTS fragments."""
        response = self.engine.generate_response(text, on_sentence=None)
        if not response:
            print("[AI] No response generated.", flush=True)
            return

        self.event_bus.emit("assistant_sentence", text=response)
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

    AIModule.on_user_message = patched_on_user_message
    AIModule._is_creator_identity_question = classmethod(patched_creator_handler)
    AIModule._format_tool_success = staticmethod(patched_format_tool_success)
    AIModule._generate_response = patched_generate_response
    VoiceModule.on_conversation_mode_set = patched_voice_on_conversation_mode_set
    VoiceModule._can_listen = patched_voice_can_listen
    AIModule._asta_runtime_patch_applied = True
