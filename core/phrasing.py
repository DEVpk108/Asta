"""Spoken lines that don't sound the same every time.

ASTA's chat shows the exact facts; the voice should sound like a person.
Each kind of moment has several ways of saying it, and the phrasebook never
reuses one of the last few it picked for that moment, so a demo run twice in
a row doesn't repeat itself word for word.
"""

from __future__ import annotations

import random
import re
import threading
from collections import deque

LINES: dict[str, list[str]] = {
    "ack": [
        "Okay sir, I'm on it.",
        "Sure, I'll take care of that and let you know when it's done.",
        "Got it. I'm working on it now.",
        "Alright, leave it with me.",
        "On it, sir. This shouldn't take long.",
        "Sure thing. Give me a moment to work through it.",
        "Okay, I'll handle that in the background. Talk to me anytime meanwhile.",
    ],
    "ack_fix": [
        "Okay sir, let me find that bug.",
        "Sure, I'll track down the problem and fix it.",
        "Got it. Let me take a look and sort it out.",
        "Alright, I'll dig into it and get it working.",
    ],
    "approve_run": [
        "I'm ready to run {command} to check the work. Should I go ahead?",
        "Before I run {command}, I need your okay.",
        "Everything's in place. Can I run {command} now?",
        "I'd like to run {command} to make sure it works. Is that alright?",
    ],
    "approve_replace": [
        "I'd like to replace {path}. Is that okay?",
        "This will overwrite {path}. Should I go ahead?",
        "Can I replace {path} with the new version?",
    ],
    "approve_test_edit": [
        "To finish this I'd have to change the test file {path}. Do you want me to?",
        "This would change the test file {path}, which you didn't ask me to touch. Should I go ahead?",
        "Heads up, the plan edits the test file {path}. Is that okay with you?",
    ],
    "approve_suffix": [
        " Say approve coding step when you're ready.",
        " Just say approve coding step, or use the button.",
        " Say approve coding step to let me continue.",
    ],
    "docker_start": [
        "{product} isn't running, so I'm starting it now. This can take a minute.",
        "Looks like {product} is closed. I'm opening it for you; give me a minute.",
        "I need {product} for this and it isn't open yet. Starting it up now.",
        "{product}'s not running. Let me open it, and then I'll carry on.",
    ],
    "image_build": [
        "The sandbox image isn't built yet, so I'm building it now. This takes a few minutes.",
        "I need to build the sandbox image first. Give me a few minutes.",
        "One-time setup: I'm building the sandbox image. It'll take a few minutes.",
    ],
    "done_tests": [
        "All done, sir. The tests pass now. Check it out.",
        "I've finished the work and the tests are passing. Take a look.",
        "Done. I fixed it and the tests pass. Have a look when you're ready.",
        "That's sorted. The tests are green now. Check it out.",
        "I've done the work. Everything passes now, have a look.",
    ],
    "done_opened": [
        "All done, sir. It's open on your screen now. Have fun!",
        "I've finished it and it's running on your screen. Check it out.",
        "Done. It's up on your desktop, give it a try.",
        "That's ready. I've opened it for you, take a look.",
    ],
    "approve_open": [
        "It's written. Can I open {path} on your desktop? It runs outside the sandbox.",
        "I'd like to open {path} on your screen so you can try it. It runs outside the sandbox. Is that okay?",
        "{path} is ready. Should I launch it on your desktop? It won't be sandboxed.",
    ],
    "plan_failed": [
        "Sorry, I couldn't put a plan together for that. The reason is on screen.",
        "I couldn't work out a safe plan for that one. Have a look at the screen for why.",
        "That one didn't come together, sir. I've put the reason on screen.",
    ],
    "done_already_passing": [
        "Good news, the tests already pass. Nothing needed changing.",
        "I checked, and the tests already pass. I didn't need to change anything.",
        "Looks like it was already working. The tests pass as they are.",
    ],
    "done": [
        "I've done the work. Check it out.",
        "All done, sir. Have a look.",
        "Finished. The changes are ready for you to check.",
        "Done. Take a look when you get a chance.",
        "That's done. Check it out.",
    ],
    "printed": [
        "It printed {output}.",
        "The output was {output}.",
        "It says {output}.",
    ],
    "tries": [
        "It took me {n} tries to get it right.",
        "The first attempt didn't work, but I got there.",
        "It needed {n} attempts, but it's working now.",
    ],
    "paused": [
        "I had to pause the task. The reason is on screen.",
        "I've paused the project task. Have a look at the screen for why.",
        "Something stopped me, so I paused the task. The details are on screen.",
    ],
}

_FIX_WORDS = re.compile(r"\b(fix|bug|broken|error|failing|fails|crash|wrong|repair|debug)\b", re.IGNORECASE)


class Phrasebook:
    def __init__(self, lines: dict[str, list[str]] | None = None, rng: random.Random | None = None,
                 memory: int = 3) -> None:
        self.lines = lines or LINES
        self.rng = rng or random.Random()
        self.memory = memory
        self._recent: dict[str, deque[str]] = {}
        self._lock = threading.Lock()

    def say(self, *kinds: str, **slots) -> str:
        """A line for this moment (several kinds = one combined pool)."""
        pool = [line for kind in kinds for line in self.lines.get(kind, [])]
        if not pool:
            return ""
        key = "+".join(kinds)
        with self._lock:
            recent = self._recent.setdefault(key, deque(maxlen=max(0, min(self.memory, len(pool) - 1))))
            fresh = [line for line in pool if line not in recent] or pool
            line = self.rng.choice(fresh)
            if recent.maxlen:
                recent.append(line)
        return line.format(**slots)


PHRASES = Phrasebook()


def say(*kinds: str, **slots) -> str:
    return PHRASES.say(*kinds, **slots)


def acknowledgement(goal: str) -> str:
    """"Okay sir, I'm on it." (or a fix-flavoured variant for bug fixes)."""
    if _FIX_WORDS.search(str(goal or "")):
        return say("ack_fix", "ack")
    return say("ack")
