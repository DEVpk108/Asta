"""Talking about the project, not just commanding it.

"Hey, I want you to make changes in the snake game we were building" names
no change; "Yeah, I'd like to add that" only makes sense after the previous
turn. Rules can't follow that, so the local model reads the sentence with
what it needs to understand it (the open project, its files, the last
project task, the last few turns) and answers with one JSON decision:

* task: the change is clear; ``goal`` is a complete instruction for the
  coding planner (file names, exact change, everything the user said);
* ask: the user wants project work but hasn't said what; ``question`` is
  one short question, and the next turn is read with it in mind;
* none: not project work (chat, questions, music, apps).

The decision only starts a normal project task: planning, approvals and
verification stay exactly as they are.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

SYSTEM_PROMPT = (
    "You decide whether the user is asking A.S.T.A., a voice assistant, to work on the code "
    "project that is open in its workspace. Reply with ONE JSON object and nothing else: "
    '{"action": "task" | "ask" | "none", "goal": "<instruction or empty>", "question": "<question or empty>"}. '
    "task: the user wants code in the project created or changed AND it is clear what to do. "
    "goal = one complete instruction for the coding agent, in the user's words: name the file(s) "
    "from the project files (or a new file name), say exactly what to change, and keep every detail "
    "the user gave (keys, names, numbers, colours). If they want to see or play the result, end "
    "with 'Then open <file> on the desktop.' "
    "ask: the user wants project work but has not said what to do yet. question = one short, "
    "friendly question about what they want, naming the file or program they mean. Never ask for "
    "something they already said; if the conversation already says what to do, use task. "
    "none: anything else (questions about code, explanations, chat, music, apps). "
    "Use the recent conversation to understand 'it', 'that', 'yes' and 'the game'."
)

_PROJECT_WORDS = re.compile(
    r"\b(?:project|code|program|programs|game|app|application|script|file|files|function|feature|"
    r"bug|workspace|window|level|score|controls?|keys?)\b",
    re.IGNORECASE,
)
_REQUEST_WORDS = re.compile(
    r"\b(?:want|wanna|need|like|make|change|changes|add|fix|update|build|create|edit|improve|"
    r"modify|write|remove|let'?s|can\s+you|could\s+you|would\s+you|please|working\s+on|"
    r"we\s+were|continue|finish)\b",
    re.IGNORECASE,
)
_FILE_TOOLS = ("filesystem.", "project.")
PENDING_SECONDS = 180.0


@dataclass(slots=True)
class ProjectDecision:
    action: str
    goal: str = ""
    question: str = ""


def parse_decision(raw: Any) -> ProjectDecision | None:
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
    action = str(data.get("action") or "").strip().lower()
    goal = " ".join(str(data.get("goal") or "").split())[:600]
    question = " ".join(str(data.get("question") or "").split())[:240]
    if action == "task" and goal:
        return ProjectDecision("task", goal=goal)
    if action == "ask" and question:
        return ProjectDecision("ask", question=question)
    if action in {"task", "ask", "none"}:
        return ProjectDecision("none")
    return None


def is_project_tool_call(names) -> bool:
    return any(str(name).startswith(_FILE_TOOLS) or name == "system.run_command" for name in names)


class ProjectConversation:
    """Per-assistant state: the question we asked, and the model call."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.pending: dict[str, Any] | None = None

    # -- when to pay for a model pass ------------------------------------------

    def pending_question(self) -> str:
        if self.pending and self.clock() < self.pending["until"]:
            return str(self.pending["question"])
        self.pending = None
        return ""

    def worth_checking(self, text: str, files: list[str], recent_task: Any = None) -> bool:
        value = str(text or "")
        if len(re.findall(r"\w+", value)) < 2:
            return False
        if self.pending_question():
            return True
        lowered = value.lower()
        stems = {PurePosixPath(item).stem.lower() for item in files if len(PurePosixPath(item).stem) >= 3}
        names = {PurePosixPath(item).name.lower() for item in files}
        if any(name in lowered for name in names) or any(re.search(rf"\b{re.escape(stem)}\b", lowered) for stem in stems):
            return True
        if not _REQUEST_WORDS.search(value):
            return False
        return bool(_PROJECT_WORDS.search(value)) or recent_task is not None

    # -- the decision ----------------------------------------------------------

    def decide(
        self,
        text: str,
        complete: Callable[..., str],
        *,
        project: str,
        files: list[str],
        recent_task: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> ProjectDecision | None:
        lines = [f"Project: {project or 'unnamed'}",
                 "Project files: " + (", ".join(files[:30]) if files else "(empty)")]
        if recent_task:
            lines.append(f"Last project task: {recent_task}")
        turns = [item for item in (history or []) if str(item.get("text") or "").strip()]
        if turns and turns[-1].get("role") == "user" and turns[-1].get("text", "").strip() == str(text).strip():
            turns = turns[:-1]
        if turns:
            lines.append("Recent conversation:")
            for item in turns[-6:]:
                who = "User" if item.get("role") == "user" else "A.S.T.A."
                lines.append(f"{who}: {' '.join(str(item.get('text')).split())[:300]}")
        question = self.pending_question()
        if question:
            lines.append(f"A.S.T.A. just asked: {question}")
        lines.append(f"User said: {text}")
        try:
            raw = complete(SYSTEM_PROMPT, "\n".join(lines), max_tokens=200)
        except Exception as exc:  # the model being down must not break chat
            print(f"[AI] Project conversation unavailable: {type(exc).__name__}: {exc}", flush=True)
            return None
        decision = parse_decision(raw)
        if decision is None:
            return None
        if decision.action == "ask":
            self.pending = {"question": decision.question, "until": self.clock() + PENDING_SECONDS}
        else:
            self.pending = None
        if decision.action == "task":
            decision.goal = ground_goal(decision.goal, text, files)
        return decision


def ground_goal(goal: str, text: str, files: list[str]) -> str:
    """Name the project file the user meant when the model's goal didn't."""
    from core.tools.workspace import mentioned_paths

    if mentioned_paths(goal):
        return goal
    said = f"{goal} {text}".lower()
    matches = [
        item for item in files
        if len(PurePosixPath(item).stem) >= 3
        and re.search(rf"\b{re.escape(PurePosixPath(item).stem.lower())}\b", said)
    ]
    if len(matches) == 1:
        return f"{goal.rstrip(' .')} (in {matches[0]})."
    return goal
