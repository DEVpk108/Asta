"""Everyday commands while a project task works in the background.

"Play Baithi Hai on Spotify" used to be refused with "A project task is still
active" even though ASTA had just said "we can keep talking while it runs".
Opening apps, searching and playing music don't touch the project, so they run
now, next to the project task, without becoming a second task.

Only plans made entirely of the tools below qualify; anything that reads or
writes files, runs code, or controls the mouse/keyboard freely still waits.
The plan is built deterministically (no model call: the coding model is busy).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from uuid import uuid4

from core.contracts import IntentType

SIDE_TOOLS = frozenset({
    "system.open_application",
    "system.launch_application",
    "browser.search",
    "media.control",
    "media.ui_play",
    "computer.wait",
    "audio.mute",
    "audio.unmute",
})


class SideActionRunner:
    def __init__(self, kernel, *, say):
        self.kernel = kernel
        self.say = say  # say(text) shows + speaks a line
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _plan(self, text, intent):
        from core.planner import Planner

        kernel = self.kernel
        planner = Planner(
            kernel.tool_registry,
            media_manager=getattr(kernel, "media_manager", None),
            application_manager=getattr(kernel, "application_manager", None),
            agent_brain=None,  # deterministic only
        )
        return planner.plan(str(getattr(intent, "normalized_text", "") or text), intent=intent)

    def try_start(self, text, intent) -> bool:
        """Run it now if it is a side action; False means "not one of those"."""
        if getattr(intent, "intent", None) is not IntentType.COMMAND:
            return False
        try:
            plan = self._plan(text, intent)
        except Exception as exc:
            print(f"[Side] Not a side action ({type(exc).__name__}: {exc})", flush=True)
            return False
        tools = [str(step.metadata.get("tool") or "") for step in plan.steps]
        if not tools or any(tool not in SIDE_TOOLS for tool in tools):
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.say("I'm still on your last request. Give me a second.")
                return True
            self._thread = threading.Thread(
                target=self._run, args=(text, plan), name="asta-side-action", daemon=True,
            )
            self._thread.start()
        print(f"[Side] Running alongside the project task: {' -> '.join(tools)}", flush=True)
        return True

    def _run(self, text, plan):
        from ai.ai_module import AIModule

        task = SimpleNamespace(
            id=f"side-{uuid4().hex[:12]}", goal=str(text), plan=plan,
            metadata={"side_action": True, "confidence": 0.98, "classifier": "rules"}, evidence=[],
        )
        runtime = self.kernel.task_runtime
        last = None
        for step in plan.steps:
            request = runtime.build_plan_request(task, step)
            if request is None:
                self.say("I couldn't do that one while the project task runs.")
                return
            request.metadata["side_action"] = True
            result = self.kernel.tool_dispatcher.dispatch(request)
            print(
                f"[Side] {request.tool}: {'ok' if result.success else result.error}",
                flush=True,
            )
            if not result.success:
                self.say(f"I couldn't do that: {result.error or 'the step failed'}")
                return
            task.evidence.append({"tool": result.tool, "success": True, "output": result.output,
                                  "plan_step_id": step.id})
            last = result
            if step.metadata.get("completion_message") and isinstance(last.metadata, dict):
                last.metadata.setdefault("completion_message", step.metadata["completion_message"])
        if last is not None:
            self.say(AIModule._format_tool_success(last))
