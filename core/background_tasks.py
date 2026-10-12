"""One background project task; ordinary conversation never owns its task slot.

The existing synchronous bus/tool loop is preserved, but project planning and
approved continuations are dispatched on a dedicated worker. Desktop actions
are held while it owns the single TaskManager slot. No arbitrary parallel tools.
"""
from __future__ import annotations

import queue
import re
import threading
from uuid import uuid4


_TERMINAL = {"completed", "failed", "cancelled"}


class BackgroundTaskCoordinator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=8)
        self.closing = threading.Event()
        self.cancelled = threading.Event()
        self.state = None
        self.workspace_identity = None
        self.running_operation = False
        self.worker = threading.Thread(target=self._run, name="asta-project-worker", daemon=True)
        self.subscriptions = {
            "workspace_updated": self.on_workspace,
            "task_created": self.on_created,
            "task_progress": self.on_progress,
            "tool_confirmation_required": self.on_approval,
            "task_completed": self.on_terminal,
            "task_failed": self.on_terminal,
            "task_cancelled": self.on_terminal,
            "task_paused": self.on_terminal,
        }
        for event, callback in self.subscriptions.items():
            kernel.event_bus.subscribe(event, callback)
        self.worker.start()

    def _say(self, text, spoken=None):
        """Show and speak a coordinator reply.

        The HUD chat and speech listen to ``assistant_sentence``; emitting only
        ``assistant_response`` left status/queue/cancel replies invisible.
        """
        if spoken is not None and spoken != text:
            self.kernel.event_bus.emit("assistant_sentence", text=text, spoken=spoken)
        else:
            self.kernel.event_bus.emit("assistant_sentence", text=text)
        self.kernel.event_bus.emit("assistant_response", text=text)

    @property
    def busy(self):
        with self.lock:
            return bool(self.running_operation or (self.state and self.state["phase"] not in _TERMINAL))

    def in_worker(self):
        return threading.current_thread() is self.worker

    def owns(self, task_id):
        with self.lock:
            return bool(task_id and self.state and self.state.get("task_id") == task_id)

    def snapshot(self):
        with self.lock:
            return dict(self.state) if self.state else None

    def _publish(self):
        snapshot = self.snapshot()
        if snapshot:
            self.kernel.event_bus.emit("background_task_changed", task=snapshot)

    def _phase(self, phase, message=""):
        with self.lock:
            if self.state is None:
                return
            if self.cancelled.is_set() and phase != "cancelled":
                return
            self.state.update(phase=phase, message=message)
        self._publish()

    def start(self, goal, intent):
        with self.lock:
            current = self.kernel.task_manager.current()
            active = current is not None and current.status.value not in _TERMINAL
            if self.closing.is_set() or self.busy or active:
                self._say("A task is already active. Please finish or cancel it before starting another project task.")
                return False
            self.cancelled.clear()
            manager = getattr(self.kernel, "workspace_manager", None)
            self.workspace_identity = manager.snapshot().get("project_path", "") if manager else None
            self.state = {"job_id": str(uuid4()), "task_id": None, "phase": "queued",
                          "goal": str(goal), "message": "Project task queued"}
            self.schedule(lambda: self._start_goal(goal, intent))
        self._publish()
        self._say(
            "I’ve queued that project task in the background. We can keep talking while it runs.",
            spoken="On it.",
        )
        return True

    def schedule(self, operation):
        with self.lock:
            if self.closing.is_set() or self.cancelled.is_set() or self.state is None:
                return False
            job_id = self.state["job_id"]
            try:
                self.queue.put_nowait((job_id, operation))
            except queue.Full:
                self._say("The project worker queue is full. Please wait for the current step.")
                return False
            return True

    def _run(self):
        while not self.closing.is_set():
            try:
                job_id, operation = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    if self.closing.is_set() or self.cancelled.is_set() or not self.state or self.state["job_id"] != job_id:
                        continue
                    self.running_operation = True
                operation()
            except Exception as exc:
                self._phase("failed", f"Project worker failed ({type(exc).__name__})")
                snapshot = self.snapshot()
                task = self.kernel.task_manager.get(snapshot["task_id"]) if snapshot and snapshot["task_id"] else None
                if task is not None and task.status.value not in _TERMINAL:
                    self.kernel.task_manager.fail("Background worker failed.", task.id)
                self._say("The background project task failed. Check its status before retrying.")
            finally:
                with self.lock:
                    self.running_operation = False
                self.queue.task_done()

    def _start_goal(self, goal, intent):
        self._phase("planning", "Preparing a safe project plan")
        task = self.kernel.task_runtime.start_goal(goal, intent)
        if self.cancelled.is_set():
            return
        if task is None:
            self._phase("failed", "No safe executable project plan was created")
            self._say("I couldn’t create a safe executable project plan. No further project actions will run.")
        elif task.status.value in _TERMINAL or task.status.value == "paused":
            self._phase(task.status.value, "Project task " + task.status.value)
        elif self.snapshot()["phase"] != "awaiting_approval":
            self._phase("working", "Project task is active")

    def on_created(self, task=None, **kwargs):
        if not isinstance(task, dict) or not self.in_worker():
            return
        with self.lock:
            if not self.state or self.state["task_id"] is not None:
                return
            self.state["task_id"] = task["id"]
            should_cancel = self.cancelled.is_set()
        if should_cancel:
            self.kernel.task_manager.cancel("Background task cancelled during planning.", task["id"])
        self._publish()

    def on_progress(self, task_id=None, phase="working", text="", **kwargs):
        if self.owns(task_id) or (self.in_worker() and self.state and not self.state["task_id"]):
            self._phase(phase, str(text))

    def on_approval(self, request=None, **kwargs):
        if request is not None and self.owns(request.metadata.get("task_id")):
            self._phase("awaiting_approval", "Review the requested action; approval is still required")

    def on_terminal(self, task=None, **kwargs):
        if isinstance(task, dict) and self.owns(task.get("id")):
            if task.get("status") in _TERMINAL:
                for pending in self.kernel.approval_manager.list_pending():
                    if pending.request.metadata.get("task_id") == task["id"]:
                        self.kernel.approval_manager.reject(pending.request.request_id)
            self._phase(task.get("status", "failed"), "Project task " + task.get("status", "failed"))
            if task.get("status") == "paused":
                self._say_paused(task)

    def _say_paused(self, task):
        """A paused task blocks new actions, so never pause silently."""
        metadata = task.get("metadata") or {}
        setup = metadata.get("capability_setup") or {}
        if isinstance(setup, dict) and setup.get("status") == "running":
            return  # the setup flow explains itself
        reason = ""
        for item in reversed(task.get("evidence") or []):
            if not isinstance(item, dict):
                continue
            check = item.get("verification") if isinstance(item.get("verification"), dict) else {}
            if item.get("success") is False and item.get("error"):
                reason = str(item["error"])
                break
            if check.get("status") in {"failed", "unknown"} and check.get("summary"):
                reason = str(check["summary"])
                break
        reason = reason.strip().rstrip(".")
        if len(reason) > 160:
            reason = reason[:157].rstrip() + "..."
        text = "I paused the project task"
        text += f" because of this: {reason}." if reason else "."
        text += " Say cancel task to stop it."
        print(f"[Tasks] Paused: {reason or 'no reason recorded'}", flush=True)
        # The reason is shown in the chat; saying it aloud is too long.
        self._say(text, spoken="I paused the project task. The reason is on screen.")

    def on_workspace(self, workspace=None, **kwargs):
        if self.busy and self.workspace_identity is not None and isinstance(workspace, dict):
            if workspace.get("project_path", "") != self.workspace_identity:
                self.cancel()

    def allow_request(self, request):
        if self.closing.is_set():
            return False
        task_id = request.metadata.get("task_id")
        task = self.kernel.task_manager.get(task_id) if task_id else None
        if task is not None and task.status.value in _TERMINAL:
            return False
        if self.owns(request.metadata.get("task_id")):
            return not (self.closing.is_set() or self.cancelled.is_set())
        return not self.busy

    def cancel(self):
        if not self.busy:
            return False
        self.cancelled.set()
        self._phase("cancelled", "Cancellation requested; any in-flight tool must finish safely")
        runtime = getattr(self.kernel, "model_runtime", None)
        if runtime is not None:
            for client in tuple(runtime.clients):
                if client.purpose == "agent":
                    client.cancel()
        snapshot = self.snapshot()
        if snapshot and snapshot["task_id"]:
            # Remove pending approvals before emitting a terminal event.
            for pending in self.kernel.approval_manager.list_pending():
                if pending.request.metadata.get("task_id") == snapshot["task_id"]:
                    self.kernel.approval_manager.reject(pending.request.request_id)
            task = self.kernel.task_manager.get(snapshot["task_id"])
            if task is not None and task.status.value not in _TERMINAL:
                self.kernel.task_manager.cancel("Background cancellation requested.", task.id)
        return True

    def handle_control(self, text):
        value = re.sub(r"[.!?]+$", "", str(text).strip().lower())
        if value in {"cancel coding task", "cancel project task", "stop coding task", "stop project task"}:
            message = "Cancellation requested. An in-flight tool may need to finish safely." if self.cancel() else "There is no active background project task."
        elif value in {"coding status", "project task status", "background task status", "what is the coding task doing"} or re.fullmatch(
            r"(?:(?:okay|ok|hey)\s+)?(?:are you|is asta)\s+(?:still\s+)?(?:generating|writing|working on)\s+(?:(?:the|my)\s+)?(?:code|project|coding task)", value):
            state = self.snapshot()
            message = (f"The project task is {state['phase'].replace('_', ' ')}. {state['message']}" if state else "There is no background project task.")
            if state and state.get("phase") == "awaiting_approval":
                message += " Say approve coding step or reject coding step."
        else:
            return False
        self._say(message)
        return True

    def shutdown(self):
        if self.closing.is_set():
            return
        self.cancel()
        self.closing.set()
        if not self.in_worker():
            self.worker.join(timeout=5)
        for event, callback in self.subscriptions.items():
            self.kernel.event_bus.unsubscribe(event, callback)
