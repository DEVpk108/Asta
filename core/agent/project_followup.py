"""Follow-up changes to the program A.S.T.A. just built.

"Okay, make the snake move with W, A, S, D also" right after "write a Snake
game" names no file and no code word, so the stateless router called it
small talk and the chat model (which has no project tools) gave up. When the
last thing A.S.T.A. did was a project task, and the request reads like a
change to that work (it names one of its files, a word of its goal, or "it"
/ "the game"), this turns it into a project task on those files.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from core.contracts import IntentResult, IntentType

_LEAD = re.compile(
    r"^(?:(?:okay|ok|alright|great|nice|cool|good|hey|asta|please|so|now|and|then|also|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|just|i\s+want\s+you\s+to|"
    r"i\s+need\s+you\s+to|i'?d\s+like\s+you\s+to)[\s,.!]+)+",
    re.IGNORECASE,
)
_CHANGE = re.compile(
    r"^(?:make|let|allow|add|change|update|edit|modify|fix|set|remove|delete|"
    r"increase|decrease|reduce|speed|slow|rename|give|support|replace|"
    r"improve|enable|disable)\b",
    re.IGNORECASE,
)
_PRONOUN = re.compile(
    r"\b(?:it|its|this|that|the\s+(?:game|program|app|code|script|window|file))\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"^(?:what|why|how|when|where|who|which|is|are|does|do|did)\b", re.IGNORECASE)
_PROJECT_TOOLS = {"filesystem.write_file", "project.open_program", "system.run_command"}
_STOP = {
    "write", "code", "python", "after", "that", "then", "with", "this", "file", "program",
    "create", "build", "make", "open", "file", "desktop", "using", "your", "please", "should",
    "script", "small", "simple", "test", "tests", "from", "into", "game", "games",
}


def _max_age() -> float:
    try:
        return max(0.0, float(os.getenv("ASTA_PROJECT_FOLLOWUP_SECONDS", "1800")))
    except ValueError:
        return 1800.0


def last_project_task(task_manager: Any, *, max_age: float | None = None):
    """The most recent task overall, if it was a project task that changed files."""
    try:
        tasks = list(task_manager.list()) if task_manager is not None else []
    except Exception:
        return None
    if not tasks:
        return None
    task = max(tasks, key=lambda t: getattr(t, "updated_at", None) or datetime.min.replace(tzinfo=timezone.utc))
    try:
        age = (datetime.now(timezone.utc) - task.updated_at).total_seconds()
    except (AttributeError, TypeError):
        return None
    if age < 0 or age > (_max_age() if max_age is None else max_age):
        return None
    if (getattr(task, "metadata", None) or {}).get("intent_type") != IntentType.TASK.value:
        return None
    steps = list(getattr(getattr(task, "plan", None), "steps", None) or ())
    if not any(str((step.metadata or {}).get("tool") or "") in _PROJECT_TOOLS for step in steps):
        return None
    return task


def _files(task) -> tuple[list[str], bool]:
    written, opened = [], False
    for step in getattr(getattr(task, "plan", None), "steps", None) or ():
        metadata = step.metadata or {}
        tool = str(metadata.get("tool") or "")
        path = str(metadata.get("path") or "").replace("\\", "/").strip()
        if tool == "project.open_program":
            opened = True
        if tool in {"filesystem.write_file", "project.open_program"} and path and path not in written:
            written.append(path)
    return written, opened


def followup_intent(text: Any, task_manager: Any) -> IntentResult | None:
    value = " ".join(str(text or "").split()).strip()
    request = _LEAD.sub("", value).strip()
    if not request or _QUESTION.match(request) or not _CHANGE.match(request):
        return None
    task = last_project_task(task_manager)
    if task is None:
        return None
    files, opened = _files(task)
    if not files:
        return None
    lowered = request.lower()
    words = set(re.findall(r"[a-z][a-z0-9_]{3,}", lowered))
    stems = {PurePosixPath(item).stem.lower() for item in files}
    goal_words = set(re.findall(r"[a-z][a-z0-9_]{3,}", str(task.goal).lower())) - _STOP
    named = {item for item in files if item.lower() in lowered or PurePosixPath(item).name.lower() in lowered}
    if not (named or words & stems or words & goal_words or _PRONOUN.search(request)):
        return None

    targets = sorted(named) or files
    names = ", ".join(targets)
    request = request.rstrip(" .!")
    goal = f"In {names}: {request[0].lower() + request[1:]}. Keep everything else in {names} working."
    if opened:
        goal += f" Then open {targets[0]} on the desktop again."
    from core.intent_router import IntentRouter

    entities = IntentRouter._extract_project_task(f"edit the file {names}") or {}
    if not entities:
        return None
    entities["goal"] = goal
    entities["followup_of"] = str(getattr(task, "id", "") or "")
    return IntentResult(
        intent=IntentType.TASK,
        confidence=0.85,
        normalized_text=goal,
        entities=entities,
        requires_tools=True,
        requires_reasoning=True,
        classifier="project_followup",
    )
