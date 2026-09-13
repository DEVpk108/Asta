import re

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
    """Add deterministic mixed-response and screenshot-opening behavior."""
    from ai.ai_module import AIModule

    if getattr(AIModule, "_asta_runtime_patch_applied", False):
        return

    original_on_user_message = AIModule.on_user_message
    original_creator_handler = AIModule._is_creator_identity_question
    original_format_tool_success = AIModule._format_tool_success

    def patched_on_user_message(self, text):
        if isinstance(text, str) and text.strip():
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

    AIModule.on_user_message = patched_on_user_message
    AIModule._is_creator_identity_question = classmethod(patched_creator_handler)
    AIModule._format_tool_success = staticmethod(patched_format_tool_success)
    AIModule._asta_runtime_patch_applied = True
