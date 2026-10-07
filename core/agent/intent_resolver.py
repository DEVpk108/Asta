"""Understand commands the rule router doesn't know, using recent work.

The rule router only knows fixed phrasings. Two things sit behind it:

1. ``resolve_screen_reference``: deterministic. "Baithi Hai is on the screen,
   can you play it" / "play this one" / "play the song on the screen" mean
   "press that result in the app I was just using".
2. ``resolve_with_llm``: the local LLM maps any other action-like sentence to
   one structured action (JSON), grounded in the most recent task, so "it",
   "this" and "the screen" resolve to real songs and apps. Its answer is
   turned into the same entities the router produces, so execution, approvals
   and verification stay deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

_SCREEN = r"(?:the\s+|my\s+|your\s+)?(?:screen|display|monitor)"
_ON_SCREEN = re.compile(
    rf"(?P<pre>.*?)\s*(?:\bis|'s|\bare)\s+(?:now\s+)?(?:on|showing\s+on|visible\s+on|there\s+on|up\s+on)\s+{_SCREEN}\b",
    re.IGNORECASE,
)
_PLAY_ON_SCREEN = re.compile(
    rf"\b(?:play|start|put\s+on)\s+(?:the\s+)?(?P<title>.*?)\s*(?:that'?s\s+|which\s+is\s+)?(?:on|from|shown\s+on|showing\s+on)\s+{_SCREEN}\b",
    re.IGNORECASE,
)
_PLAY_DEMONSTRATIVE = re.compile(
    # "play this one" / "play that song please" -- but not "play This Is America".
    r"\b(?:play|start)\s+(?:this|that)(?:\s+(?:one|song|track))?(?:\s+(?:please|now|for\s+me))?\s*(?:[.!?,]|$)",
    re.IGNORECASE,
)
_PLAY_WORD = re.compile(r"\b(?:play|start|put\s+on|chala\w*|baja\w*)\b", re.IGNORECASE)
_LEADS = re.compile(
    r"^.*\b(?:now|and|so|then|okay|ok|see|look|here|because|but)\b\s+", re.IGNORECASE
)
_FILLER_TITLE = re.compile(
    r"^(?:it|this|that|everything|something|the\s+(?:song|track|result|results|list|one|page)|"
    r"song|track|results?|one|first\s+one|the\s+first\s+one|first\s+song|the\s+first\s+song)$",
    re.IGNORECASE,
)


def _clean_title(text: str) -> str:
    title = _LEADS.sub("", str(text or "").strip())
    title = re.sub(r"^(?:the\s+)?(?:song|track)\s+", "", title, flags=re.IGNORECASE)
    title = title.strip(" ,.'\"!?")
    if not title or _FILLER_TITLE.match(title) or len(title.split()) > 8:
        return ""
    return title


def screen_title(text: str) -> tuple[bool, str]:
    """(refers to something on the screen, title it names if any)."""
    sentences = [s for s in re.split(r"[.!?]+\s*", str(text or "")) if s.strip()]
    for sentence in sentences:
        match = _ON_SCREEN.search(sentence)
        if match:
            return True, _clean_title(match.group("pre"))
    for sentence in sentences:
        match = _PLAY_ON_SCREEN.search(sentence)
        if match:
            return True, _clean_title(match.group("title"))
    if _PLAY_DEMONSTRATIVE.search(str(text or "")):
        return True, ""
    return False, ""


def resolve_screen_reference(text: str, work: Any) -> dict[str, Any] | None:
    """Entities for "play the song that's on the screen" in the recent app."""
    refers, title = screen_title(text)
    if not refers or not _PLAY_WORD.search(str(text or "")):
        # "Baithi Hai is on the screen" alone doesn't ask for anything yet;
        # the LLM resolver (with the recent task) decides that one.
        return None
    if work is None or not (work.provider or work.application):
        return None
    query = title or str(work.query or "").strip()
    if not query:
        return None
    entities: dict[str, Any] = {"action": "media", "operation": "play", "query": query, "on_screen": True}
    if work.provider:
        entities["provider"] = work.provider
    return entities


# -- LLM resolver ------------------------------------------------------

ACTIONS = ("play_on_screen", "play", "pause", "resume", "next", "previous", "open", "close", "search", "retry", "none")

SYSTEM_PROMPT = (
    "You turn one spoken sentence into one action for a desktop voice assistant. "
    "The speech recognizer makes spelling mistakes; fix obvious ones in song and app names. "
    "Reply with a single JSON object and nothing else: "
    '{"action": "<one of: ' + ", ".join(ACTIONS) + '>", "query": "<song, artist or search text, or empty>", '
    '"app": "<app name, or empty>"}. '
    "play_on_screen = play a song/video the user says is already showing in the app. "
    "play = find and play a song or artist. pause/resume/next/previous = control what is playing. "
    "open/close = open or close an app. search = search the web. retry = repeat the recent action. "
    "none = a question, chat, or anything that is not a clear command. "
    "Use the recent action to resolve 'it', 'this', 'that' and 'the screen'. "
    "Only use apps the user or the recent action named. If unsure, use none."
)


@dataclass(slots=True)
class Resolution:
    action: str
    query: str = ""
    app: str = ""


def context_note(work: Any, last_app: str = "") -> str:
    lines = []
    if work is not None:
        outcome = "failed" if work.failed else (work.status or "finished")
        line = f"Recent action: {work.goal} ({outcome})"
        if work.query:
            line += f"; song/query: {work.query}"
        if work.application or work.provider:
            line += f"; app: {work.application or work.provider}"
        lines.append(line + ".")
    if last_app:
        lines.append(f"Last opened app: {last_app}.")
    return "\n".join(lines) or "No recent action."


def parse_resolution(raw: str) -> Resolution | None:
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    action = str(data.get("action") or "").strip().lower().replace(" ", "_")
    if action not in ACTIONS:
        return None
    return Resolution(
        action=action,
        query=str(data.get("query") or "").strip(" .'\""),
        app=str(data.get("app") or "").strip(" .'\""),
    )


def resolution_entities(
    resolution: Resolution,
    work: Any,
    provider_for: Callable[[str], str | None] | None = None,
) -> dict[str, Any] | None:
    """Router-shaped entities for a resolution; None = not a usable command."""
    action = resolution.action
    app = resolution.app
    query = resolution.query
    provider = ""
    if app and provider_for is not None:
        try:
            provider = str(provider_for(app) or "")
        except Exception:
            provider = ""
    if not provider and work is not None and (not app or app.lower() in {
        str(work.application or "").lower(), str(work.provider or "").lower()
    }):
        provider = str(work.provider or "")
    if action == "play_on_screen":
        query = query or str(getattr(work, "query", "") or "")
        if not query or not (provider or app):
            return None
        entities: dict[str, Any] = {"action": "media", "operation": "play", "query": query, "on_screen": True}
        if provider or app:
            entities["provider"] = provider or app
        return entities
    if action == "play":
        if not query:
            return None
        entities = {"action": "media", "operation": "play", "query": query}
        if provider or app:
            entities["provider"] = provider or app
        return entities
    if action in {"pause", "resume", "next", "previous"}:
        return {"action": "media", "operation": "play" if action == "resume" else action}
    if action in {"open", "close"}:
        return {"action": action, "target": app} if app else None
    if action == "search":
        return {"action": "search", "query": query} if query else None
    return None


def resolve_with_llm(
    text: str,
    complete: Callable[[str, str], str],
    *,
    work: Any = None,
    last_app: str = "",
) -> Resolution | None:
    user = f"{context_note(work, last_app)}\nUser said: {text}"
    try:
        raw = complete(SYSTEM_PROMPT, user)
    except Exception as exc:  # the LLM being down must not break chat
        print(f"[AI] Intent resolver unavailable: {exc}", flush=True)
        return None
    return parse_resolution(raw)
