"""Handle tool calls that the chat model writes into its reply.

LFM2-family models answer some requests with native tool-call markup such as
``<|tool_call_start|>[media.control(action='pause')]<|tool_call_end|>``
instead of prose. That text must never be spoken or shown in chat. A small
allowlist of low-risk, reversible calls is routed through the normal command
path (same tools, same permissions); anything else is dropped with a short
spoken fallback.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

TOOL_CALL_BLOCK = re.compile(r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|$)", re.S)
_SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")
_BARE_CALL = re.compile(r"^\s*\[?\s*[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\s*\(", re.I)

MEDIA_OPERATIONS = {
    "play": "play", "resume": "play", "pause": "pause", "toggle": "toggle",
    "next": "next", "skip": "next", "previous": "previous", "stop": "pause",
}
FALLBACK_TEXT = "Sorry, I couldn't do that. Please try saying it again."


def looks_like_tool_markup(text: str) -> bool:
    value = str(text or "")
    return "<|tool_call" in value or bool(_BARE_CALL.match(value))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _parse_call_list(source: str) -> list[tuple[str, dict[str, Any]]]:
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError:
        return []
    body = tree.body
    nodes = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            continue
        name = _call_name(node.func)
        if not name:
            continue
        kwargs: dict[str, Any] = {}
        try:
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ValueError
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            continue
        calls.append((name, kwargs))
    return calls


def split_tool_calls(text: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Return (spoken text without markup, parsed calls)."""
    value = str(text or "")
    calls: list[tuple[str, dict[str, Any]]] = []
    for match in TOOL_CALL_BLOCK.finditer(value):
        calls.extend(_parse_call_list(match.group(1)))
    clean = TOOL_CALL_BLOCK.sub(" ", value)
    if not calls and _BARE_CALL.match(clean):
        calls = _parse_call_list(clean)
        if calls:
            clean = ""
    clean = " ".join(_SPECIAL_TOKEN.sub(" ", clean).split())
    return clean, calls


def native_tool_call_text(tool_calls: list[dict[str, Any]]) -> str:
    """Render OpenAI-style streamed tool_calls in the same markup form."""
    parts = []
    for call in tool_calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        rendered = ", ".join(f"{key}={value!r}" for key, value in args.items() if str(key).isidentifier())
        parts.append(f"{name}({rendered})")
    return f"<|tool_call_start|>[{', '.join(parts)}]<|tool_call_end|>" if parts else ""


def media_entities(calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any] | None:
    """Map an allowed model call to command entities, or None."""
    if len(calls) != 1:
        return None
    name, kwargs = calls[0]
    if name not in {"media.control", "media_control"}:
        return None
    raw = str(kwargs.get("operation") or kwargs.get("action") or "").strip().lower()
    operation = MEDIA_OPERATIONS.get(raw)
    if not operation:
        return None
    entities: dict[str, Any] = {"action": "media", "operation": operation}
    provider = kwargs.get("provider")
    if isinstance(provider, str) and provider.strip():
        entities["provider"] = provider.strip().lower()
    return entities


def speech_filter(on_sentence):
    """Wrap a sentence callback so tool-call markup is never spoken."""
    def filtered(sentence):
        if not sentence:
            return
        if looks_like_tool_markup(sentence):
            clean, _calls = split_tool_calls(sentence)
            if not clean:
                return
            sentence = clean
        on_sentence(sentence)
    return filtered


def finish_model_response(module, response: str) -> str | None:
    """Route model tool calls; return the text to show, or None if handled."""
    clean, calls = split_tool_calls(response)
    if not calls and not looks_like_tool_markup(response):
        return response
    _forget_markup(module, clean)
    entities = media_entities(calls)
    if entities is not None:
        from core.contracts import IntentResult, IntentType

        print(f"[AI] Model tool call routed: media.control {entities['operation']}", flush=True)
        module._handle_command_intent(
            IntentResult(
                intent=IntentType.COMMAND,
                confidence=0.9,
                normalized_text=clean or entities["operation"],
                entities=entities,
                requires_tools=True,
                classifier="llm_tool_call",
            )
        )
        return None
    names = ", ".join(name for name, _ in calls) or "unparsed"
    print(f"[AI] Ignored model tool call ({names}); not on the allowlist.", flush=True)
    if clean:
        return clean
    if any(name.startswith(("filesystem.", "project.")) or name == "system.run_command" for name, _ in calls):
        # The chat model can't touch the project; the request is understood
        # (and started as a project task, or asked about) instead.
        handler = getattr(module, "_project_request_from_tool_call", None)
        if callable(handler) and handler():
            return None
    module.event_bus.emit("assistant_sentence", text=FALLBACK_TEXT)
    return FALLBACK_TEXT


def _forget_markup(module, clean: str) -> None:
    """Keep raw markup out of replayed history so the model doesn't copy it."""
    engine = getattr(module, "engine", None)
    holders = [engine, getattr(engine, "engine", None), getattr(engine, "_engine", None)]
    holders.extend((getattr(engine, "engines", None) or {}).values())
    for holder in holders:
        # Plain engines keep `_messages`; the model router keeps `history`.
        for attribute in ("_messages", "history"):
            messages = getattr(holder, attribute, None)
            if not isinstance(messages, list) or not messages:
                continue
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") == "assistant" and looks_like_tool_markup(str(last.get("content") or "")):
                messages[-1] = {"role": "assistant", "content": clean or "(Handled the request.)"}
