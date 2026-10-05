"""ASTA's awareness of its own recent work.

Keeps "try again", "what happened?" and "why didn't it play?" grounded in the
task that actually ran (goal, status, failing step, error), instead of letting
the LLM guess from a two-word message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.transliteration import DEVANAGARI_TO_ENGLISH, romanize_hinglish

_LEAD = r"(?:(?:okay|ok|hey|asta|please|so|now|and|then|can\s+you|could\s+you|will\s+you|just)\s+)*"
_RETRY = re.compile(
    rf"^{_LEAD}(?:"
    r"try(?:\s+(?:it|that|this))?\s+(?:again|once\s+more|one\s+more\s+time)|retry(?:\s+(?:it|that))?|"
    r"(?:do|play|run)\s+(?:it|that)\s+again|again|one\s+more\s+time|once\s+more|"
    r"(?:fir|phir)\s+se(?:\s+(?:try|koshish))?(?:\s+(?:karo|kar|karna|kijiye|kar\s+do))?|"
    r"dobara(?:\s+(?:try|koshish))?(?:\s+(?:karo|kar|karna|kijiye|kar\s+do))?|"
    r"ek\s+(?:baar|bar)\s+(?:aur|or|phir)(?:\s+(?:try|koshish))?(?:\s+(?:karo|kar))?|"
    r"try\s+(?:karo|kar|again\s+karo)"
    r")(?:\s+(?:please|sir|na|yaar))?$",
    re.IGNORECASE,
)
_STATUS = re.compile(
    rf"^{_LEAD}(?:"
    r"what\s+happened|what(?:'s|\s+is)\s+(?:wrong|the\s+problem|the\s+issue|happening)|what\s+went\s+wrong|"
    r"why\s+(?:did\s+(?:it|that)\s+fail|didn'?t\s+(?:it|that|you)\b.*|did\s+(?:it|that)\s+not\b.*|"
    r"(?:is|isn'?t)\s+(?:it|that)\b.*|(?:it|that)\s+(?:failed|didn'?t\b.*))|"
    r"did\s+(?:it|that)\s+(?:work|fail)|what\s+are\s+you\s+doing|what\s+were\s+you\s+doing|"
    r"kya\s+hua|kya\s+problem\s+hai|kya\s+kar\s+rahe\s+ho"
    r")\W*$",
    re.IGNORECASE,
)

# A retry/status question only refers to work this recent.
MAX_AGE_SECONDS = 600


_WORDS = {"ट्राई": "try", "ट्राय": "try", "अगेन": "again", "अगैन": "again", "रिट्राई": "retry",
          "दोबारा": "dobara", "प्लीज": "please", "प्लीज़": "please", "ओके": "okay"}


def _roman(text: str) -> str:
    """Devanagari -> English/Hinglish words ("ट्राई अगेन" -> "try again")."""
    out = []
    for token in re.findall(r"[\u0900-\u097F]+|[^\s\u0900-\u097F]+", str(text or "")):
        if re.match(r"[\u0900-\u097F]", token):
            out.append(_WORDS.get(token) or DEVANAGARI_TO_ENGLISH.get(token) or romanize_hinglish(token))
        else:
            out.append(token)
    value = " ".join(out).lower()
    value = re.sub(r"[^\w\s']", " ", value)
    return " ".join(value.split())


def is_retry_request(text: str) -> bool:
    return bool(_RETRY.match(_roman(text)))


def is_status_question(text: str) -> bool:
    return bool(_STATUS.match(_roman(text)))


@dataclass(slots=True)
class WorkSummary:
    goal: str
    status: str
    failed_step: str | None
    error: str | None
    age_seconds: float

    @property
    def failed(self) -> bool:
        return self.status == "failed" or bool(self.failed_step)


def recent_work(task_manager: Any, *, max_age: float = MAX_AGE_SECONDS) -> WorkSummary | None:
    """Most recent task with its outcome, if it is recent enough to refer to."""
    if task_manager is None:
        return None
    try:
        tasks = list(task_manager.list())
    except Exception:
        return None
    if not tasks:
        return None
    task = max(tasks, key=lambda t: getattr(t, "updated_at", None) or datetime.min.replace(tzinfo=timezone.utc))
    updated = getattr(task, "updated_at", None)
    try:
        age = (datetime.now(timezone.utc) - updated).total_seconds()
    except TypeError:
        return None
    if age < 0 or age > max_age:
        return None
    goal = str(getattr(task, "goal", "") or "").strip()
    if not goal:
        return None
    status = str(getattr(getattr(task, "status", None), "value", getattr(task, "status", "")) or "")
    failed_step = None
    error = str(getattr(task, "error", "") or "").strip() or None
    plan = getattr(task, "plan", None)
    for step in getattr(plan, "steps", None) or ():
        if str(getattr(step.status, "value", step.status)) == "failed":
            failed_step = step.description
            error = error or str((step.metadata or {}).get("error") or "").strip() or None
    return WorkSummary(goal, status, failed_step, error, age)


def _plain_error(error: str | None) -> str:
    text = str(error or "").strip().rstrip(".")
    text = re.sub(r"\s*\((?:window title|request_id)[^)]*\)", "", text)
    return text[:160]


def describe(work: WorkSummary) -> str:
    """Spoken account of the last task."""
    goal = work.goal.rstrip(".?!")
    if work.status in {"active", "pending", "running"}:
        return f"I'm still working on it: {goal}."
    if work.failed:
        reason = _plain_error(work.error)
        because = f" The problem was: {reason}." if reason else ""
        return f"I tried to {goal}, but it didn't work.{because} Say 'try again' and I'll retry."
    if work.status == "cancelled":
        return f"I stopped the last task: {goal}."
    return f"My last task was to {goal}, and it finished."


def llm_context(work: WorkSummary | None) -> str | None:
    """Short note for the LLM so follow-ups refer to what actually happened."""
    if work is None or work.age_seconds > 180:
        return None
    outcome = "failed" if work.failed else (work.status or "finished")
    reason = _plain_error(work.error) if work.failed else ""
    note = f"Your most recent action: '{work.goal}' ({outcome}"
    note += f": {reason})." if reason else ")."
    return note + " Only describe actions using this record; do not claim new actions."
