"""Deterministic presentation/screenshot shortcuts owned by native AI routing."""
import re
from core.contracts import IntentResult, IntentType
from .post_response_capture import extract_prompt as _extract_post_response_screenshot

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
    "show latest screen shot",
    "show the latest screen shot",
    "open latest screen shot",
    "open the latest screen shot",
}

_SCREENSHOT_CAPTURE_PATTERN = re.compile(
    r"\b(?:take|capture|grab|snap|make)\b.*\b(?:screenshot|screen\s*shot|screen\s*capture)\b",
    re.IGNORECASE,
)

_CONVERSATION_LEAD_PATTERN = re.compile(
    r"^(?:okay|ok|please)\s*[,;:.!?-]*\s+",
    re.IGNORECASE,
)


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


def handle_shortcut(ai, text):
    if not isinstance(text, str) or not text.strip() or _extract_post_response_screenshot(text):
        return False
    normalized = " ".join(text.strip().split())
    conversation = _CONVERSATION_LEAD_PATTERN.sub("", normalized, count=1)
    if conversation != normalized and ai._handle_conversation_mode_command(conversation):
        return True
    opened = _is_screenshot_open_request(text)
    if not opened:
        routed = ai.kernel.intent_router.analyze(text)
        commands = (routed.entities or {}).get("commands")
        if routed.intent == IntentType.COMMAND and isinstance(commands, list) and len(commands) >= 2:
            return False
        if not _is_screenshot_capture_request(text):
            return False
    action = "open_screenshot" if opened else "screenshot"
    intent = IntentResult(intent=IntentType.COMMAND, confidence=1.0,
        normalized_text="open latest screenshot" if opened else "take a screenshot",
        entities={"action": action}, requires_tools=True, classifier="native_shortcut")
    try:
        request = ai.tool_request_builder.build(intent)
    except ValueError as exc:
        ai._emit_assistant_text(f"I couldn't {'open' if opened else 'take'} the screenshot: {exc}")
        return True
    request.metadata["screenshot_open_request" if opened else "direct_screenshot_request"] = True
    ai.event_bus.emit("tool_request", request=request)
    return True
