from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any
import os
import threading

from .contracts import (
    IntentResult,
    IntentType,
    PlanStatus,
    PlanStepStatus,
    TaskStatus,
    ToolRequest,
    ToolResult,
    PlanStep,
)
from .tools.request_builder import ToolRequestBuilder
from .planner import PlanningError
from .module import Module



from core.tools.approval import APPROVAL_EXPIRED_ERROR
from core.tools.workspace import (  # noqa: E402
    existing_workspace_file,
    is_test_path,
    mentioned_paths,
    text_change_summary,
)
from core.tools.sandbox import (  # noqa: E402
    SANDBOX_SETUP_COMMAND,
    SANDBOX_SETUP_MESSAGE,
    is_sandbox_setup_error,
)
from core.media.request import UNSUPPORTED_MUSIC_APPS  # noqa: E402

class TaskRuntimeModule(Module):
    """Bind user goals and tool activity to the active AgentTask.

    The module listens on the existing EventBus so task orchestration stays
    independent from the LLM and from individual tool implementations.
    """

    def __init__(self, kernel):
        super().__init__(
            name="TaskRuntimeModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)
        self._incremental_sessions: dict[str, dict[str, Any]] = {}
        self._incremental_session_lock = threading.RLock()
        kernel.task_runtime = self

    def initialize(self):
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe(
            "incremental_user_message",
            self.on_incremental_user_message,
        )
        self.event_bus.subscribe("tool_request", self.on_tool_request)
        self.event_bus.subscribe("tool_result", self.on_tool_result)
        self.event_bus.subscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        self.event_bus.subscribe(
            "capability_setup_completed",
            self.on_capability_setup_completed,
        )
        self.event_bus.subscribe(
            "incremental_voice_session_started",
            self.on_incremental_voice_session_started,
        )
        self.event_bus.subscribe(
            "incremental_voice_session_finished",
            self.on_incremental_voice_session_finished,
        )
        self.event_bus.subscribe(
            "task_completed",
            self.on_task_completed,
        )
        self.event_bus.subscribe(
            "task_failed",
            self.on_task_failed,
        )
        self.event_bus.subscribe(
            "task_cancelled",
            self.on_task_cancelled,
        )
        print("[Tasks] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe(
            "incremental_user_message",
            self.on_incremental_user_message,
        )
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe("tool_result", self.on_tool_result)
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        self.event_bus.unsubscribe(
            "capability_setup_completed",
            self.on_capability_setup_completed,
        )
        self.event_bus.unsubscribe(
            "incremental_voice_session_started",
            self.on_incremental_voice_session_started,
        )
        self.event_bus.unsubscribe(
            "incremental_voice_session_finished",
            self.on_incremental_voice_session_finished,
        )
        self.event_bus.unsubscribe(
            "task_completed",
            self.on_task_completed,
        )
        self.event_bus.unsubscribe(
            "task_failed",
            self.on_task_failed,
        )
        self.event_bus.unsubscribe(
            "task_cancelled",
            self.on_task_cancelled,
        )
        print("[Tasks] Stopped", flush=True)

    def on_user_message(self, text):
        self._handle_user_message(text, acknowledge=True)

    def on_incremental_voice_session_started(self, session_id=None, *args, **kwargs):
        value = str(session_id or "").strip()
        if not value:
            return
        with self._incremental_session_lock:
            self._incremental_sessions[value] = {
                "closed": False,
                "active_task_id": None,
                "pending": deque(),
                "next_segment": 0,
            }
        print(
            f"[Tasks] Incremental session started: {value}",
            flush=True,
        )

    def on_incremental_voice_session_finished(
        self,
        session_id=None,
        *args,
        **kwargs,
    ):
        value = str(session_id or "").strip()
        if not value:
            return
        with self._incremental_session_lock:
            state = self._incremental_sessions.get(value)
            if state is None:
                state = {
                    "closed": True,
                    "active_task_id": None,
                    "pending": deque(),
                    "next_segment": 0,
                }
                self._incremental_sessions[value] = state
            else:
                state["closed"] = True
            self._cleanup_incremental_session_locked(value)
        print(
            f"[Tasks] Incremental session finished: {value}",
            flush=True,
        )

    def on_incremental_user_message(self, text, *args, **kwargs):
        # Incremental commands are already being heard and acted on while the
        # user is speaking. Do not inject a spoken acknowledgment into the
        # same live utterance.
        session_id = str(kwargs.get("session_id") or "").strip()
        commit_id = str(kwargs.get("commit_id") or "").strip() or None
        if not session_id:
            self._handle_user_message(text, acknowledge=False)
            return

        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and background.busy and not background.in_worker():
            return
        value = str(text or "").strip()
        if not value:
            return

        with self._incremental_session_lock:
            state = self._incremental_sessions.setdefault(
                session_id,
                {
                    "closed": False,
                    "active_task_id": None,
                    "pending": deque(),
                    "next_segment": 0,
                },
            )
            active_task_id = state.get("active_task_id")

        active_task = (
            self.kernel.task_manager.get(active_task_id)
            if active_task_id
            else None
        )
        if active_task is not None and active_task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            with self._incremental_session_lock:
                state["pending"].append((value, commit_id))
            self.event_bus.emit(
                "task_progress",
                task_id=active_task.id,
                phase="queued",
                text="I heard the next part. I’ll continue after this step.",
            )
            print(
                f"[Tasks] Queued incremental continuation for "
                f"{active_task.id}: {value}",
                flush=True,
            )
            return

        self._start_incremental_segment(
            value,
            session_id=session_id,
            commit_id=commit_id,
        )

    def _handle_user_message(self, text, *, acknowledge=True):
        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and background.busy and not background.in_worker():
            return
        if not text:
            return

        intent = self.kernel.intent_router.analyze(text)
        if intent.intent is not IntentType.COMMAND:
            return

        if acknowledge:
            acknowledgment = self._acknowledgment_for_intent(intent)
            if acknowledgment:
                self.event_bus.emit(
                    "command_acknowledged",
                    text=acknowledgment,
                    intent=intent.to_dict() if hasattr(intent, "to_dict") else None,
                )
                self.event_bus.emit(
                    "assistant_sentence",
                    text=acknowledgment,
                )

        self.start_plan(
            intent.normalized_text,
            intent,
        )

    def _start_incremental_segment(
        self,
        text: str,
        *,
        session_id: str,
        commit_id: str | None = None,
    ):
        intent = self.kernel.intent_router.analyze(text)
        if intent.intent is not IntentType.COMMAND:
            return None

        with self._incremental_session_lock:
            state = self._incremental_sessions.setdefault(
                session_id,
                {
                    "closed": False,
                    "active_task_id": None,
                    "pending": deque(),
                    "next_segment": 0,
                },
            )
            state["next_segment"] += 1
            segment_index = state["next_segment"]

        # Build the task first without emitting its first tool request. Some
        # tool runtimes execute synchronously, so the session must register the
        # task as active before any tool_result/task_completed event can arrive.
        task = self.start_plan(
            intent.normalized_text,
            intent,
            incremental_session_id=session_id,
            incremental_commit_id=commit_id,
            incremental_segment_index=segment_index,
        )
        if task is not None:
            with self._incremental_session_lock:
                state = self._incremental_sessions.get(session_id)
                if state is not None:
                    state["active_task_id"] = task.id

            next_step = self._next_ready_plan_step(task)
            if next_step is not None:
                request = self.build_plan_request(task, next_step)
                if request is not None:
                    print(
                        f"[Tasks] Incremental segment ready: "
                        f"{next_step.id} -> {request.tool}",
                        flush=True,
                    )
                    self.event_bus.emit(
                        "tool_request",
                        request=request,
                    )
        return task

    def start_plan(
        self,
        goal: str,
        intent: IntentResult,
        *,
        incremental_session_id: str | None = None,
        incremental_commit_id: str | None = None,
        incremental_segment_index: int | None = None,
    ):
        """Create an executable task plan from any structured command intent."""
        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and background.busy and not background.in_worker():
            return None
        self._emit_progress(
            "planning",
            "I’m working out how to do that.",
        )

        agent_brain = getattr(self.kernel, "agent_brain", None)
        agent_thinking = bool(
            agent_brain is not None
            and getattr(agent_brain, "enabled", False)
        )
        if agent_thinking:
            self.event_bus.emit(
                "agent_thinking_started",
                phase="planning",
            )

        try:
            plan = self.kernel.planner.plan(
                goal,
                intent=intent,
            )
        except PlanningError as exc:
            print(f"[Tasks] Planning failed: {exc}", flush=True)
            return None
        finally:
            if agent_thinking:
                self.event_bus.emit(
                    "agent_thinking_finished",
                    phase="planning",
                )

        pending_steps = [step.description for step in plan.steps]

        metadata = {
            "intent_type": intent.intent.value,
            "confidence": intent.confidence,
            "classifier": intent.classifier,
            "intent_entities": dict(intent.entities),
            "replan_attempts": 0,
        }
        if incremental_session_id:
            metadata["incremental_session"] = {
                "session_id": incremental_session_id,
                "commit_id": incremental_commit_id,
                "segment_index": incremental_segment_index,
            }

        if plan.metadata.get("agent_mode") == "cognitive_v1":
            metadata["agent_state"] = {
                "goal": plan.metadata.get("agent_goal_summary", goal),
                "success_conditions": list(
                    plan.metadata.get("agent_success_conditions") or ()
                ),
                "beliefs": {},
                "observations": [],
                "actions": [],
                "decision_history": [],
                "current_strategy": str(
                    plan.metadata.get("agent_rationale") or ""
                ),
                "uncertainty": float(
                    (
                        0.5
                        if plan.metadata.get("agent_uncertainty") is None
                        else plan.metadata.get("agent_uncertainty")
                    )
                ),
            }

        task = self.kernel.task_manager.create(
            intent.normalized_text,
            pending_steps=pending_steps,
            plan=plan,
            metadata=metadata,
        )

        print(
            f"[Tasks] Started task {task.id}: {task.goal}",
            flush=True,
        )
        self.event_bus.emit(
            "task_progress",
            task_id=task.id,
            phase="planned",
            text="I know what I need to do. Starting now.",
        )
        return task

    def start_goal(self, goal: str, intent: IntentResult):
        """Start a scoped, open-ended task and dispatch its first planned step."""
        if not isinstance(intent, IntentResult) or intent.intent is not IntentType.TASK:
            raise ValueError("start_goal requires a task intent")

        task = self.start_plan(goal, intent)
        if task is None or task.plan is None:
            return None

        next_step = self._next_ready_plan_step(task)
        if next_step is None:
            self.kernel.task_manager.fail(
                "The task plan has no executable first step.",
                task.id,
            )
            return None

        request = self.build_plan_request(task, next_step)
        if request is None:
            self.kernel.task_manager.fail(
                f"Unable to build a tool request for plan step '{next_step.id}'.",
                task.id,
            )
            return None

        self.event_bus.emit("tool_request", request=request)
        return task

    def on_task_completed(self, task=None, *args, **kwargs):
        self._advance_incremental_session_from_terminal(
            task,
            terminal_state=TaskStatus.COMPLETED,
        )

    def on_task_failed(self, task=None, *args, **kwargs):
        self._advance_incremental_session_from_terminal(
            task,
            terminal_state=TaskStatus.FAILED,
        )

    def on_task_cancelled(self, task=None, *args, **kwargs):
        self._advance_incremental_session_from_terminal(
            task,
            terminal_state=TaskStatus.CANCELLED,
        )

    def _advance_incremental_session_from_terminal(self, task, *, terminal_state):
        if isinstance(task, dict):
            task_id = str(task.get("id") or "").strip()
            metadata = task.get("metadata") or {}
        else:
            task_id = str(getattr(task, "id", "") or "").strip()
            metadata = getattr(task, "metadata", {}) or {}

        session = metadata.get("incremental_session")
        session_id = (
            str(session.get("session_id") or "").strip()
            if isinstance(session, dict)
            else ""
        )
        if not task_id or not session_id:
            return

        with self._incremental_session_lock:
            state = self._incremental_sessions.get(session_id)
            if state is None or state.get("active_task_id") != task_id:
                return

            pending = state.get("pending")
            if not pending:
                state["active_task_id"] = None
                self._cleanup_incremental_session_locked(session_id)
                return

            if terminal_state is not TaskStatus.COMPLETED:
                state["active_task_id"] = None
                pending.clear()
                self._cleanup_incremental_session_locked(session_id)
                queued_count = 0
            else:
                next_text, next_commit_id = pending.popleft()
                state["active_task_id"] = None
                queued_count = len(pending) + 1

        if terminal_state is not TaskStatus.COMPLETED:
            print(
                f"[Tasks] Incremental session {session_id} stopped after "
                f"{terminal_state.value} task {task_id}; "
                f"discarded {queued_count} queued continuation(s).",
                flush=True,
            )
            self.event_bus.emit(
                "task_progress",
                task_id=task_id,
                phase="recovery",
                text=(
                    "That step did not complete, so I’m holding the remaining "
                    "spoken steps rather than running them out of order."
                ),
            )
            return

        print(
            f"[Tasks] Advancing incremental session {session_id}: "
            f"continuing with {next_text!r}",
            flush=True,
        )
        self._start_incremental_segment(
            next_text,
            session_id=session_id,
            commit_id=next_commit_id,
        )

    def _cleanup_incremental_session_locked(self, session_id: str) -> None:
        state = self._incremental_sessions.get(session_id)
        if state is None or not state.get("closed"):
            return
        if state.get("active_task_id") is not None:
            return
        pending = state.get("pending")
        if pending:
            return
        self._incremental_sessions.pop(session_id, None)

    def on_tool_request(self, request):
        background = getattr(self.kernel, "background_tasks", None)
        if background is not None and background.busy and not background.in_worker() and not background.owns(getattr(request, "metadata", {}).get("task_id")):
            return
        if not isinstance(request, ToolRequest):
            return

        task = self.kernel.task_manager.current()
        if task is None or task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return

        task_step = self._request_step(request)
        plan_step_id = request.metadata.get("plan_step_id")
        if not isinstance(plan_step_id, str) or not plan_step_id.strip():
            plan_step_id = self._request_plan_step_id(request, task)
        request.metadata["task_id"] = task.id
        request.metadata["task_step"] = task_step
        if plan_step_id is not None:
            request.metadata["plan_step_id"] = plan_step_id
            self.kernel.task_manager.set_plan_step_status(
                plan_step_id,
                PlanStepStatus.RUNNING,
                task.id,
            )
        self.kernel.task_manager.set_step(task_step, task.id)
        self.event_bus.emit(
            "task_progress",
            task_id=task.id,
            phase="executing",
            text=self._progress_for_request(request),
        )

    def on_tool_result(self, result):
        if isinstance(result, ToolResult) and result.metadata.get("background_conflict"):
            return
        if not isinstance(result, ToolResult):
            return

        task_id = result.metadata.get("task_id")
        task = self.kernel.task_manager.get(task_id) if task_id else self.kernel.task_manager.current()
        if task is None or task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return

        self._remember_opened_application(result)
        self._record_agent_observation(task, result)

        evidence: dict[str, Any] = {
            "type": "tool_result",
            "tool": result.tool,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "request_id": result.metadata.get("request_id"),
        }
        plan_step_id = result.metadata.get("plan_step_id")
        if plan_step_id:
            evidence["plan_step_id"] = plan_step_id
        if (
            result.success
            and result.tool == "filesystem.write_file"
            and isinstance(result.output, dict)
            and result.output.get("diff")
        ):
            print(
                f"[Tasks] Changed {result.output.get('path')} "
                f"(+{result.output.get('lines_added', 0)} -{result.output.get('lines_removed', 0)}):\n"
                + str(result.output["diff"]).rstrip(),
                flush=True,
            )
        if not result.success:
            if result.error in {
                "Tool execution rejected by user.",
                APPROVAL_EXPIRED_ERROR,
            }:
                if plan_step_id:
                    self.kernel.task_manager.set_plan_step_status(
                        plan_step_id,
                        PlanStepStatus.FAILED,
                        task.id,
                    )
                self.kernel.task_manager.add_evidence(evidence, task.id)
                self.kernel.task_manager.fail(
                    "The approval expired, so the step was not run."
                    if result.error == APPROVAL_EXPIRED_ERROR
                    else "The user denied the requested tool action.",
                    task.id,
                )
                return

        prefix_soft_failure = False
        if not result.success:
            step_metadata = self._plan_step_metadata(task, plan_step_id)
            if self._has_later_project_steps(task, step_metadata):
                # A mixed request ("open Spotify and play X, then write
                # main.py"): the music part failed, but the project part is
                # independent. Report it and continue instead of replanning
                # the whole task around the music app.
                prefix_soft_failure = True
                note = self._unconfirmed_prefix_note(step_metadata)
                evidence["verification"] = {
                    "status": "unconfirmed",
                    "verified": False,
                    "summary": str(result.error or "The step did not complete."),
                    "note": note,
                }
                print(
                    f"[Tasks] {note} ({result.error or 'step failed'}) "
                    "Continuing with the project steps.",
                    flush=True,
                )
                self.event_bus.emit(
                    "task_progress",
                    task_id=task.id,
                    phase="working",
                    text=f"{note} Continuing with the project steps.",
                )

        if (
            not result.success
            and not prefix_soft_failure
            and result.tool == "filesystem.write_file"
            and str(result.error or "").startswith("The file already exists.")
        ):
            # The plan wrote a file that is already in the workspace (often
            # from an earlier run of the same request).
            existing = self._existing_file_write(task, plan_step_id)
            if existing is not None and existing[0] == "unchanged":
                output = existing[1]
                print(
                    f"[Tasks] {output['path']} already has the requested content; "
                    "nothing to write.",
                    flush=True,
                )
                result = ToolResult(
                    success=True,
                    tool=result.tool,
                    output=output,
                    duration_seconds=result.duration_seconds,
                    metadata=result.metadata,
                )
                evidence.update(success=True, output=output, error=None)
            elif existing is not None:
                # Different content: replacing the user's file needs a yes.
                overwrite_request = existing[1]
                confirm_path = overwrite_request.metadata.get("confirm_overwrite")
                evidence["overwrite_confirmation"] = {
                    "path": confirm_path or overwrite_request.arguments.get("path"),
                    "line_endings_only": not confirm_path,
                }
                self.kernel.task_manager.add_evidence(evidence, task.id)
                print(
                    f"[Tasks] {confirm_path} already exists with different content; "
                    "asking before replacing it."
                    if confirm_path
                    else f"[Tasks] {overwrite_request.arguments.get('path')} has the same text "
                    "with different line endings; rewriting it.",
                    flush=True,
                )
                self.event_bus.emit("tool_request", request=overwrite_request)
                return

        if not result.success and not prefix_soft_failure:
            print(
                f"[Tasks] Step failed: {result.tool or 'tool'}: "
                f"{result.error or 'no error message'}",
                flush=True,
            )
            if result.tool == "system.run_command" and is_sandbox_setup_error(result.error):
                # Docker/Podman is stopped or the local image is missing.
                # Replanning the user's code cannot fix that, so repair the
                # environment instead (once per task), then rerun the step.
                if self._start_sandbox_repair(task, plan_step_id, result, evidence):
                    return
                print(
                    "[Tasks] Sandbox setup needed: start Docker Desktop, then run "
                    f"`{SANDBOX_SETUP_COMMAND}` in the ASTA folder.",
                    flush=True,
                )
                if plan_step_id:
                    self.kernel.task_manager.set_plan_step_status(
                        plan_step_id,
                        PlanStepStatus.FAILED,
                        task.id,
                    )
                self.kernel.task_manager.add_evidence(evidence, task.id)
                self.kernel.task_manager.fail(SANDBOX_SETUP_MESSAGE, task.id)
                return
            self.event_bus.emit(
                "task_progress",
                task_id=task.id,
                phase="recovery",
                text=(
                    "I hit a problem with that step. "
                    "I’m figuring out what went wrong."
                ),
            )
            if plan_step_id:
                self.kernel.task_manager.set_plan_step_status(
                    plan_step_id,
                    PlanStepStatus.FAILED,
                    task.id,
                )

            # Decide before recording the current failure so RecoveryManager
            # counts only prior failures as history and the current result as
            # the next attempt. This keeps direct recovery decisions and the
            # runtime integration on the same attempt-counting contract.
            decision = self.kernel.recovery_manager.decide(
                task,
                result,
                step_id=plan_step_id,
            )
            evidence["recovery"] = decision.to_dict()

            diagnosis = None
            if decision.action.value == "replan":
                diagnosis_engine = getattr(self.kernel, "diagnosis_engine", None)
                if diagnosis_engine is not None:
                    diagnosis = diagnosis_engine.diagnose(
                        task,
                        result,
                        recovery=decision,
                        step_id=plan_step_id,
                    )
                    evidence["diagnosis"] = diagnosis.to_dict()
                    self.event_bus.emit(
                        "task_diagnosed",
                        task_id=task.id,
                        diagnosis=diagnosis.to_dict(),
                    )

            if decision.action.value == "retry":
                retry_step = (
                    task.plan.get_step(plan_step_id)
                    if task.plan is not None and plan_step_id
                    else None
                )
                if retry_step is not None:
                    retry_step.status = PlanStepStatus.READY
                    retry_request = self.build_plan_request(
                        task,
                        retry_step,
                    )
                    if retry_request is not None:
                        self.kernel.task_manager.add_evidence(
                            evidence,
                            task.id,
                        )
                        self.event_bus.emit(
                            "task_recovery_required",
                            task_id=task.id,
                            decision=decision.to_dict(),
                            diagnosis=None,
                        )
                        print(
                            f"[Tasks] Recovery retry: {retry_step.id} -> "
                            f"{retry_request.tool}",
                            flush=True,
                        )
                        self.event_bus.emit(
                            "tool_request",
                            request=retry_request,
                        )
                        return

            if decision.action.value == "wait_for_user":
                diagnosis = None
                diagnosis_engine = getattr(self.kernel, "diagnosis_engine", None)
                if diagnosis_engine is not None:
                    try:
                        diagnosis = diagnosis_engine.diagnose(
                            task,
                            result,
                            recovery=decision,
                            step_id=plan_step_id,
                        )
                        evidence["diagnosis"] = diagnosis.to_dict()
                        self.event_bus.emit(
                            "task_diagnosed",
                            task_id=task.id,
                            diagnosis=diagnosis.to_dict(),
                        )
                    except Exception as exc:
                        evidence["diagnosis_error"] = str(exc)

                setup_manager = getattr(
                    self.kernel,
                    "capability_setup_manager",
                    None,
                )
                category = (
                    getattr(
                        getattr(diagnosis, "category", None),
                        "value",
                        "",
                    )
                    .strip()
                    .lower()
                    if diagnosis is not None
                    else ""
                )
                if category == "setup_required" and setup_manager is not None:
                    provider = self._setup_provider_for_step(
                        task,
                        plan_step_id,
                        result,
                    )
                    if provider and setup_manager.supports(provider):
                        self.kernel.task_manager.pause(task.id)
                        started = setup_manager.start(
                            task,
                            capability=provider,
                            step_id=plan_step_id,
                        )
                        evidence["capability_setup"] = {
                            "capability": provider,
                            "started": bool(started),
                            "step_id": plan_step_id,
                        }
                        if started:
                            task.metadata["capability_setup"] = {
                                "capability": provider,
                                "step_id": plan_step_id,
                                "status": "running",
                            }
                            self.kernel.task_manager.add_evidence(
                                evidence,
                                task.id,
                            )
                            self.event_bus.emit(
                                "task_recovery_required",
                                task_id=task.id,
                                decision=decision.to_dict(),
                                diagnosis=diagnosis.to_dict(),
                            )
                            return

                        evidence["capability_setup"]["status"] = "unavailable"
                        self.kernel.task_manager.add_evidence(
                            evidence,
                            task.id,
                        )
                        self.event_bus.emit(
                            "task_recovery_required",
                            task_id=task.id,
                            decision=decision.to_dict(),
                            diagnosis=diagnosis.to_dict(),
                        )
                        return

                self.kernel.task_manager.add_evidence(
                    evidence,
                    task.id,
                )
                self.event_bus.emit(
                    "task_recovery_required",
                    task_id=task.id,
                    decision=decision.to_dict(),
                    diagnosis=diagnosis.to_dict() if diagnosis is not None else None,
                )
                self.kernel.task_manager.pause(task.id)
                return

            if decision.action.value == "replan":
                replan_engine = getattr(self.kernel, "replan_engine", None)
                if replan_engine is None or diagnosis is None:
                    self.kernel.task_manager.pause(task.id)
                    return

                step = (
                    task.plan.get_step(plan_step_id)
                    if task.plan is not None and plan_step_id
                    else None
                )
                replan_decision = replan_engine.choose(
                    diagnosis,
                    plan_step=step,
                )
                evidence["replan"] = replan_decision.to_dict()
                self.event_bus.emit(
                    "task_recovery_required",
                    task_id=task.id,
                    decision=decision.to_dict(),
                    diagnosis=diagnosis.to_dict(),
                )
                self.event_bus.emit(
                    "task_progress",
                    task_id=task.id,
                    phase="replanning",
                    text=(
                        "I understand the problem. "
                        "I’m trying a different way."
                    ),
                )

                attempts = int(task.metadata.get("replan_attempts", 0)) + 1
                task.metadata["replan_attempts"] = attempts
                max_replans = 3

                if replan_decision.strategy.value == "wait_for_user":
                    self.kernel.task_manager.add_evidence(evidence, task.id)
                    self.event_bus.emit(
                        "task_replan_required",
                        task_id=task.id,
                        decision=replan_decision.to_dict(),
                    )
                    self.kernel.task_manager.pause(task.id)
                    return

                if replan_decision.strategy.value == "fail" or attempts > max_replans:
                    evidence["replan_guard"] = {
                        "max_replans": max_replans,
                        "attempt": attempts,
                    }
                    self.kernel.task_manager.add_evidence(evidence, task.id)
                    self.event_bus.emit(
                        "task_replan_required",
                        task_id=task.id,
                        decision=replan_decision.to_dict(),
                    )
                    self.kernel.task_manager.fail(
                        "Autonomous replanning exhausted its safe recovery budget.",
                        task.id,
                    )
                    return

                # Record the failed attempt first: the repair planner must
                # see its output (a failing test's traceback, for example).
                self.kernel.task_manager.add_evidence(evidence, task.id)
                try:
                    new_plan = self.kernel.planner.replan(
                        task,
                        diagnosis,
                        replan_decision.strategy,
                    )
                except PlanningError as exc:
                    evidence["replan_error"] = str(exc)
                    if task.evidence and isinstance(task.evidence[-1], dict):
                        task.evidence[-1]["replan_error"] = str(exc)
                    self.event_bus.emit(
                        "task_replan_required",
                        task_id=task.id,
                        decision=replan_decision.to_dict(),
                        error=str(exc),
                    )
                    self.kernel.task_manager.pause(task.id)
                    return

                task.plan = new_plan
                task.pending_steps = [step.description for step in new_plan.steps]
                task.current_step = None
                task.error = None
                self.event_bus.emit(
                    "task_replanned",
                    task_id=task.id,
                    decision=replan_decision.to_dict(),
                    plan=new_plan.to_dict(),
                )

                # The task remains ACTIVE throughout an autonomous replan;
                # only its structured plan is replaced.
                task.plan.status = PlanStatus.ACTIVE
                task.plan.mark_ready_steps()

                next_step = self._next_ready_plan_step(task)
                if next_step is None:
                    self.kernel.task_manager.fail(
                        "Replanned task has no executable ready step.",
                        task.id,
                    )
                    return

                next_request = self.build_plan_request(task, next_step)
                if next_request is None:
                    self.kernel.task_manager.fail(
                        f"Unable to build a tool request for replanned step '{next_step.id}'.",
                        task.id,
                    )
                    return

                print(
                    f"[Tasks] Autonomous replan: {next_step.id} -> "
                    f"{next_request.tool}",
                    flush=True,
                )
                self.event_bus.emit("tool_request", request=next_request)
                return

            self.kernel.task_manager.fail(
                result.error or "Tool execution failed.",
                task.id,
            )
            return

        step = result.metadata.get("task_step") or task.current_step or result.tool
        structured_step = (
            task.plan.get_step(plan_step_id)
            if task.plan is not None and plan_step_id
            else None
        )

        # A successful tool call is not automatically a successful real-world
        # outcome for capabilities that declare an external verifier.
        verification_key = (
            str((structured_step.metadata or {}).get("verification") or "").strip()
            if structured_step is not None
            else ""
        )
        verification_engine = getattr(self.kernel, "verification_engine", None)
        if (
            verification_key
            and structured_step is not None
            and verification_engine is None
        ):
            evidence["verification"] = {
                "status": "unknown",
                "summary": "The requested verifier is unavailable.",
                "source": "guard",
                "task_id": task.id,
                "step_id": structured_step.id,
            }
            self.kernel.task_manager.set_plan_step_status(
                structured_step.id,
                PlanStepStatus.BLOCKED,
                task.id,
            )
            self.kernel.task_manager.add_evidence(evidence, task.id)
            self.event_bus.emit(
                "task_verification_required",
                task_id=task.id,
                verification=evidence["verification"],
            )
            self.kernel.task_manager.pause(task.id)
            return
        if verification_key and verification_engine is not None and structured_step is not None:
            verification = verification_engine.verify(
                task,
                structured_step,
                result,
            )
            evidence["verification"] = verification.to_dict()

            if verification.status.value == "failed":
                self.kernel.task_manager.set_plan_step_status(
                    structured_step.id,
                    PlanStepStatus.FAILED,
                    task.id,
                )
                self.kernel.task_manager.add_evidence(evidence, task.id)
                self.event_bus.emit(
                    "task_verification_failed",
                    task_id=task.id,
                    verification=verification.to_dict(),
                )
                synthetic_failure = ToolResult(
                    success=False,
                    tool=result.tool,
                    error=verification.summary,
                    metadata={
                        "task_id": task.id,
                        "task_step": step,
                        "plan_step_id": structured_step.id,
                        "verification_failure": True,
                    },
                )
                self.event_bus.emit("tool_result", result=synthetic_failure)
                return

            if verification_key and verification.status.value == "unknown":
                self.kernel.task_manager.set_plan_step_status(
                    structured_step.id,
                    PlanStepStatus.BLOCKED,
                    task.id,
                )
                self.kernel.task_manager.add_evidence(evidence, task.id)
                self.event_bus.emit(
                    "task_verification_required",
                    task_id=task.id,
                    verification=verification.to_dict(),
                )
                self.kernel.task_manager.pause(task.id)
                return

        # vision.inspect is a verification action in the deterministic
        # command/runtime paths. A successful model call with verified=false
        # is still a failed task outcome; never advance the plan as if the
        # visual condition was confirmed.
        if result.tool == "vision.inspect" and result.success:
            output = result.output if isinstance(result.output, dict) else {}
            step_metadata = self._plan_step_metadata(task, plan_step_id)
            if not bool(output.get("verified")) and self._has_later_project_steps(
                task, step_metadata
            ):
                # A mixed request ("play X, then write main.py"): the music
                # check is unconfirmed, but the project part is independent.
                # Continue and report it, instead of dropping the project.
                summary = str(
                    output.get("summary")
                    or "Visual inspection did not confirm the requested condition."
                ).strip()
                note = self._unconfirmed_prefix_note(step_metadata)
                evidence["verification"] = {
                    "status": "unconfirmed",
                    "verified": False,
                    "summary": summary,
                    "note": note,
                }
                print(f"[Tasks] {note} Continuing with the project steps.", flush=True)
                self.event_bus.emit(
                    "task_progress",
                    task_id=task.id,
                    phase="working",
                    text=f"{note} Continuing with the project steps.",
                )
            elif not bool(output.get("verified")):
                summary = str(
                    output.get("summary")
                    or "Visual inspection did not confirm the requested condition."
                ).strip()
                if plan_step_id:
                    self.kernel.task_manager.set_plan_step_status(
                        plan_step_id,
                        PlanStepStatus.FAILED,
                        task.id,
                    )
                evidence["verification"] = {
                    "status": "failed",
                    "verified": False,
                    "summary": summary,
                }
                self.kernel.task_manager.add_evidence(evidence, task.id)
                self.kernel.task_manager.fail(
                    summary,
                    task.id,
                )
                return
        if self._complete_from_verified_visual_result(
            task,
            result,
            evidence=evidence,
            plan_step_id=plan_step_id,
            step_description=step,
        ):
            return

        if self._run_agent_decision(
            task,
            result,
            evidence=evidence,
            plan_step_id=plan_step_id,
            step_description=step,
        ):
            return

        self.kernel.task_manager.complete_step(step, task.id)

        # Preserve successful tool results in the task journal as well as
        # failures. Recovery relies on the evidence stream as its history.
        self.kernel.task_manager.add_evidence(evidence, task.id)

        if plan_step_id:
            self.kernel.task_manager.set_plan_step_status(
                plan_step_id,
                PlanStepStatus.COMPLETED,
                task.id,
            )
            self.kernel.task_manager.refresh_ready_plan_steps(task.id)

        refreshed = self.kernel.task_manager.get(task.id)
        if refreshed is None:
            return

        if not refreshed.pending_steps:
            final_result = {
                "tool": result.tool,
                "output": result.output,
            }
            if refreshed.metadata.get("intent_type") == IntentType.TASK.value:
                report = self._project_task_report(refreshed)
                if not report["verified"]:
                    self.kernel.task_manager.fail(
                        "The project task lacks verified evidence for its planned outcome.",
                        refreshed.id,
                    )
                    return
                final_result.update(report)
                self.event_bus.emit(
                    "assistant_sentence",
                    text=report["message"],
                    spoken=report.get("spoken") or report["message"],
                )
            self.kernel.task_manager.complete(
                result=final_result,
                task_id=task.id,
            )
            return

        next_step = self._next_ready_plan_step(refreshed)
        if next_step is None:
            return

        try:
            next_request = self.build_plan_request(refreshed, next_step)
        except Exception as exc:
            # Never leave a task hanging silently on a bug in a step.
            print(
                f"[Tasks] Could not build plan step {next_step.id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            next_request = None
        if next_request is None:
            self.kernel.task_manager.fail(
                f"Unable to build a tool request for plan step '{next_step.id}'.",
                refreshed.id,
            )
            return

        print(
            f"[Tasks] Advancing plan: {next_step.id} -> {next_request.tool}",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=next_request)

    @staticmethod
    def _project_task_report(task) -> dict[str, Any]:
        plan = getattr(task, "plan", None)
        steps = list(getattr(plan, "steps", ()) or ())
        verification_steps = [
            step for step in steps if step.metadata.get("verification")
        ]
        verification_evidence = {
            item.get("plan_step_id"): item.get("verification", {})
            for item in (getattr(task, "evidence", ()) or ())
            if isinstance(item, dict) and item.get("plan_step_id")
        }
        if verification_steps:
            verified = all(
                isinstance(verification_evidence.get(step.id), dict)
                and verification_evidence[step.id].get("status") == "verified"
                for step in verification_steps
            )
        else:
            verified = bool(steps) and all(
                getattr(step.status, "value", "") == "completed"
                for step in steps
            )

        command_evidence = None
        for item in getattr(task, "evidence", ()) or ():
            if not isinstance(item, dict) or item.get("success") is not True:
                continue
            if item.get("tool") == "system.run_command":
                output = item.get("output")
                if isinstance(output, dict):
                    command_evidence = (item, output)

        changes = TaskRuntimeModule._project_changes(task)
        artifacts = [{"path": change["path"]} for change in changes]

        command = None
        if command_evidence is not None:
            evidence, output = command_evidence
            try:
                command_step = plan.get_step(evidence.get("plan_step_id"))
            except (AttributeError, KeyError, TypeError):
                command_step = None
            command = {
                "target": (
                    command_step.metadata.get("target")
                    if command_step is not None
                    else None
                ),
                "returncode": output.get("returncode"),
                "stdout": output.get("stdout"),
                "stderr": output.get("stderr"),
            }

        edited = TaskRuntimeModule._describe_edits(changes)
        if edited and command is not None:
            observed = str(command.get("stdout") or "").strip()
            message = (
                f"Done. {edited}, verified the result, and ran the project command "
                f"(exit {command.get('returncode')}); "
                + (f"output: {observed[:1200]}" if observed else "it printed no output.")
            )
        elif edited:
            message = f"Done. {edited} and verified the result."
        elif artifacts and command is not None:
            observed = str(command.get("stdout") or "").strip()
            path = artifacts[-1]["path"]
            message = (
                f"Done. Verified `{path}` and ran the project command "
                f"(exit {command.get('returncode')}); output: {observed[:1200]}"
            )
        elif artifacts:
            message = f"Done. Verified the project file `{artifacts[-1]['path']}`."
        elif command is not None:
            observed = str(command.get("stdout") or "").strip()
            message = (
                f"Done. Verified the project command (exit "
                f"{command.get('returncode')}); output: {observed[:1200]}"
            )
        else:
            message = (
                "Done. The planned read-only project inspection completed successfully."
                if not verification_steps
                else "Done. The planned project outcome passed its verification."
            )

        # Mixed requests: say what happened with the leading app/music part.
        prefix_notes: list[str] = []
        for item in getattr(task, "evidence", ()) or ():
            if not isinstance(item, dict):
                continue
            check = item.get("verification")
            output = item.get("output") if isinstance(item.get("output"), dict) else {}
            note = ""
            if isinstance(check, dict) and check.get("status") == "unconfirmed" and check.get("note"):
                note = str(check["note"])
            elif item.get("tool") == "vision.inspect" and output.get("source") == "media_session" and output.get("track"):
                note = f"{output['track']} is playing."
            elif item.get("tool") == "media.ui_play" and item.get("success") and output.get("verified"):
                note = str(output.get("message") or "").strip()
            if note and note not in prefix_notes:
                prefix_notes.append(note)
        attempts = int(task.metadata.get("replan_attempts", 0) or 0)
        if attempts and verified and (edited or command is not None):
            message = message.rstrip()
            message = (message if message.endswith((".", "!", "?")) else message + ".") + (
                f" It took {attempts + 1} attempts; I used the failed check's output to fix it."
            )
        if prefix_notes:
            message = " ".join(prefix_notes) + " " + message

        spoken = TaskRuntimeModule._spoken_report(
            changes, command, command_step_arguments=TaskRuntimeModule._command_words(task, command_evidence),
            attempts=attempts + 1, prefix_notes=prefix_notes,
        )

        return {
            "verified": verified,
            "spoken": spoken,
            "artifacts": artifacts,
            "changes": changes,
            "attempts": attempts + 1,
            "command": command,
            "verifications": [
                verification_evidence.get(step.id)
                for step in verification_steps
            ],
            "message": message,
        }

    @staticmethod
    def _project_changes(task) -> list[dict[str, Any]]:
        """Net change per written file: what it held before the task vs. now.

        Uses the file's first read in this task as "before" and its last
        read-back as "after", so several fix attempts report one diff.
        """
        evidence = [item for item in (getattr(task, "evidence", ()) or ()) if isinstance(item, dict)]

        def norm(value):
            return str(value or "").replace("\\", "/").strip("/")

        order: list[str] = []
        writes: dict[str, list[tuple[int, dict]]] = {}
        for index, item in enumerate(evidence):
            output = item.get("output")
            if (
                item.get("tool") == "filesystem.write_file"
                and item.get("success") is True
                and isinstance(output, dict)
                and isinstance(output.get("path"), str)
            ):
                path = norm(output["path"])
                if path not in writes:
                    order.append(path)
                writes.setdefault(path, []).append((index, output))

        changes = []
        for path in order:
            entries = writes[path]
            first_index, first = entries[0]
            last_index, last = entries[-1]
            reads = [
                (index, item["output"].get("content"))
                for index, item in enumerate(evidence)
                if item.get("tool") == "filesystem.read_file"
                and item.get("success") is True
                and isinstance(item.get("output"), dict)
                and norm(item["output"].get("path")) == path
                and isinstance(item["output"].get("content"), str)
            ]
            before = next((text for index, text in reversed(reads) if index < first_index), None)
            after = next((text for index, text in reversed(reads) if index > last_index), None)
            created = bool(first.get("created"))
            if all(output.get("unchanged") for _, output in entries):
                change = {"lines_added": 0, "lines_removed": 0, "diff": "", "diff_truncated": False}
            elif before is not None and after is not None and not created:
                change = text_change_summary(before, after, path)
            elif len(entries) == 1 and "lines_added" in last:
                change = {
                    key: last.get(key)
                    for key in ("lines_added", "lines_removed", "diff", "diff_truncated")
                }
            else:
                change = {}
            changes.append({
                "path": path,
                "created": created,
                "unchanged": all(output.get("unchanged") for _, output in entries),
                "writes": len(entries),
                **change,
            })
        return changes

    @staticmethod
    def _command_words(task, command_evidence) -> list[str]:
        if command_evidence is None:
            return []
        evidence, _ = command_evidence
        try:
            step = task.plan.get_step(evidence.get("plan_step_id"))
        except (AttributeError, KeyError, TypeError):
            return []
        target = str(step.metadata.get("target") or "").replace("\\", "/").rsplit("/", 1)[-1]
        return [target, *[str(item) for item in (step.metadata.get("arguments") or [])]]

    @staticmethod
    def _spoken_report(changes, command, *, command_step_arguments, attempts, prefix_notes) -> str:
        """The few words worth hearing; the chat keeps the full report."""
        updated = [c["path"] for c in changes if not c.get("created") and not c.get("unchanged")]
        created = [c["path"] for c in changes if c.get("created") and not c.get("unchanged")]
        did = []
        if updated:
            did.append("I updated " + " and ".join(updated))
        if created:
            did.append("I created " + " and ".join(created))
        outcome = ""
        if command is not None:
            words = " ".join(command_step_arguments).lower()
            output = str(command.get("stdout") or "").strip()
            if "test" in words or "pytest" in words:
                outcome = "the tests passed"
            elif output and "\n" not in output and len(output) <= 80:
                outcome = f"it printed {output}"
            else:
                outcome = "it ran successfully"
        if did and outcome:
            text = f"Done. {', '.join(did)}, and {outcome}."
        elif did:
            text = f"Done. {', '.join(did)}."
        elif outcome:
            text = f"Done. {outcome[0].upper() + outcome[1:]}."
        else:
            text = "Done."
        if attempts > 1:
            text += f" It took {attempts} tries."
        if prefix_notes:
            text = " ".join(prefix_notes) + " " + text
        return text

    @staticmethod
    def _describe_edits(changes) -> str:
        """ "Updated `calc.py` (+1 -1 lines)" for edits of files that existed."""
        parts = []
        for change in changes:
            if change.get("created") or change.get("unchanged"):
                continue
            stats = ""
            if isinstance(change.get("lines_added"), int):
                stats = f" (+{change['lines_added']} -{change.get('lines_removed', 0)} lines)"
            parts.append(f"`{change['path']}`{stats}")
        if not parts:
            return ""
        created = [f"`{c['path']}`" for c in changes if c.get("created") and not c.get("unchanged")]
        text = "Updated " + ", ".join(parts)
        if created:
            text += ", created " + ", ".join(created)
        return text

    def _existing_file_write(self, task, plan_step_id):
        """Resolve a create-only write that hit an existing workspace file.

        Returns ("unchanged", output) when the file already holds exactly the
        planned content, ("overwrite", request) for a one-time, approval-gated
        replacement bound to the file's current SHA-256, or None.
        """
        plan = getattr(task, "plan", None)
        if plan is None or not plan_step_id:
            return None
        try:
            step = plan.get_step(plan_step_id)
        except (KeyError, AttributeError, TypeError):
            return None
        metadata = step.metadata or {}
        if metadata.get("expected_sha256") or metadata.get("expected_sha256_from_step"):
            return None
        path = metadata.get("path")
        content = metadata.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return None
        state = existing_workspace_file(
            getattr(self.kernel, "workspace_manager", None),
            path,
        )
        if state is None:
            return None
        digest, data = state
        if data == content.encode("utf-8"):
            return "unchanged", {
                "path": path,
                "created": False,
                "unchanged": True,
                "bytes_written": 0,
                "sha256": digest,
            }
        asked = task.metadata.setdefault("overwrite_confirmations", [])
        if plan_step_id in asked:
            return None
        request = self.build_plan_request(task, step)
        if request is None:
            return None
        asked.append(plan_step_id)
        # Same text, only Windows (CRLF) line endings differ: rewriting it
        # loses nothing, so no approval is needed; the read-back check
        # still compares the exact planned content.
        same_text = data.replace(b"\r\n", b"\n") == content.encode("utf-8").replace(b"\r\n", b"\n")
        metadata = dict(request.metadata)
        if not same_text:
            metadata["confirm_overwrite"] = path
        return "overwrite", ToolRequest(
            tool=request.tool,
            arguments={**request.arguments, "expected_sha256": digest},
            request_id=f"{request.request_id}-overwrite",
            timeout_seconds=request.timeout_seconds,
            metadata=metadata,
        )

    @staticmethod
    def _plan_step_metadata(task, plan_step_id) -> dict[str, Any]:
        plan = getattr(task, "plan", None)
        if plan is None or not plan_step_id:
            return {}
        try:
            return dict(plan.get_step(plan_step_id).metadata or {})
        except (KeyError, AttributeError, TypeError):
            return {}

    @staticmethod
    def _has_later_project_steps(task, step_metadata) -> bool:
        """True for a leading user-request step followed by planned project work."""
        if step_metadata.get("user_request_prefix") is not True:
            return False
        plan = getattr(task, "plan", None)
        return any(
            step.metadata.get("user_request_prefix") is not True
            for step in (getattr(plan, "steps", ()) or ())
        )

    @staticmethod
    def _unconfirmed_prefix_note(step_metadata) -> str:
        query = str(
            step_metadata.get("media_query") or step_metadata.get("query") or ""
        ).strip()
        app = str(
            step_metadata.get("media_app") or step_metadata.get("application") or ""
        ).strip()
        if query and app:
            return f"I couldn't confirm that {query} is playing in {app}."
        return "I couldn't confirm the first part of the request."

    def _complete_from_verified_visual_result(
        self,
        task,
        result: ToolResult,
        *,
        evidence: dict[str, Any],
        plan_step_id: str | None,
        step_description: str,
    ) -> bool:
        """Complete a visual verification goal without another LLM call."""
        if result.tool != "vision.inspect" or not result.success:
            return False

        output = result.output if isinstance(result.output, dict) else {}
        if not bool(output.get("verified")):
            return False
        if self._has_later_project_steps(
            task, self._plan_step_metadata(task, plan_step_id)
        ):
            return False  # the music check is not the goal of a mixed task

        agent_state = task.metadata.get("agent_state")
        if not isinstance(agent_state, dict):
            return False

        conditions = agent_state.get("success_conditions") or []
        text = " ".join(str(item) for item in conditions).lower()
        visual_goal = any(
            marker in text
            for marker in (
                "visible",
                "visually",
                "screenshot",
                "screen",
                "window",
                "display",
                "ui",
                "interface",
            )
        )
        if not visual_goal:
            return False

        self.kernel.task_manager.complete_step(step_description, task.id)
        if plan_step_id:
            self.kernel.task_manager.set_plan_step_status(
                plan_step_id,
                PlanStepStatus.COMPLETED,
                task.id,
            )

        evidence["agent_verified_visual"] = True
        self.kernel.task_manager.add_evidence(evidence, task.id)
        self.event_bus.emit(
            "task_goal_verified",
            task_id=task.id,
            verification={
                "tool": result.tool,
                "verified": True,
                "confidence": output.get("confidence"),
                "summary": output.get("summary"),
            },
        )
        self.kernel.task_manager.complete(
            result={
                "tool": result.tool,
                "output": result.output,
                "verified": True,
            },
            task_id=task.id,
        )
        print(
            "[Agent] Visual verifier confirmed the goal; "
            "skipping redundant post-action LLM decision.",
            flush=True,
        )
        return True

    def _run_agent_decision(
        self,
        task,
        result: ToolResult,
        *,
        evidence: dict[str, Any],
        plan_step_id: str | None,
        step_description: str,
    ) -> bool:
        """Run one bounded cognitive decision after a successful tool result."""
        if self._plan_step_metadata(task, plan_step_id).get("user_request_prefix") is True:
            # Deterministic steps from the user's own words: just follow them.
            return False
        agent_state = task.metadata.get("agent_state")
        brain = getattr(self.kernel, "agent_brain", None)
        if (
            not isinstance(agent_state, dict)
            or task.plan is None
            or brain is None
            or not getattr(brain, "enabled", False)
        ):
            return False

        state_payload = {
            "goal": task.goal,
            "status": task.status.value,
            "success_conditions": list(
                agent_state.get("success_conditions") or []
            ),
            "beliefs": dict(agent_state.get("beliefs") or {}),
            "observations": list(agent_state.get("observations") or []),
            "actions": list(agent_state.get("actions") or []),
            "current_strategy": agent_state.get("current_strategy", ""),
            "uncertainty": agent_state.get("uncertainty", 1.0),
            "current_step": {
                "id": plan_step_id,
                "description": step_description,
            },
            "latest_result": {
                "tool": result.tool,
                "success": bool(result.success),
                "output": result.output,
                "error": result.error,
            },
        }

        if self._should_follow_planned_grounded_action(
            task,
            result,
            plan_step_id,
        ):
            print(
                "[Agent] Following planned grounded computer action without "
                "an extra post-action LLM call.",
                flush=True,
            )
            return False

        # When the cognitive plan already contains a direct semantic
        # visual-verification step, follow that bounded plan step instead of
        # paying for another full LLM decision immediately after the launch.
        # The specialized vision verifier remains the next observation, and
        # the agent can reason again after its result arrives.
        if self._should_follow_planned_visual_verification(
            task,
            plan_step_id,
        ):
            print(
                "[Agent] Following planned visual verification step without "
                "an extra post-action LLM call.",
                flush=True,
            )
            return False

        skip_reason = self._post_action_skip_reason(
            task,
            result,
            plan_step_id,
            evidence=evidence,
        )
        if skip_reason:
            print(
                f"[Agent] {skip_reason}; following the plan without a "
                "post-action LLM call.",
                flush=True,
            )
            return False

        self.event_bus.emit(
            "agent_thinking_started",
            task_id=task.id,
            phase="post_action_decision",
        )
        try:
            decision = brain.decide(state_payload)
        except Exception as exc:
            print(
                f"[Agent] Post-action decision unavailable; "
                f"continuing deterministic runtime: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            evidence["agent_decision_error"] = str(exc)
            self.event_bus.emit(
                "agent_thinking_finished",
                task_id=task.id,
                phase="post_action_decision",
            )
            return False

        self.event_bus.emit(
            "agent_thinking_finished",
            task_id=task.id,
            phase="post_action_decision",
        )
        decision_payload = decision.to_dict()
        evidence["agent_decision"] = decision_payload
        self._apply_agent_decision(task, decision_payload)
        self.event_bus.emit(
            "task_agent_decision",
            task_id=task.id,
            decision=decision_payload,
        )

        print(
            f"[Agent] Decision: {decision.rationale}",
            flush=True,
        )
        print(
            f"[Agent] Confidence: {decision.confidence:.2f}",
            flush=True,
        )

        if decision.needs_user:
            self.kernel.task_manager.complete_step(step_description, task.id)
            if plan_step_id:
                self.kernel.task_manager.set_plan_step_status(
                    plan_step_id,
                    PlanStepStatus.COMPLETED,
                    task.id,
                )
            self.kernel.task_manager.add_evidence(
                evidence,
                task.id,
            )
            self.event_bus.emit(
                "task_agent_wait_required",
                task_id=task.id,
                decision=decision_payload,
            )
            self.kernel.task_manager.pause(task.id)
            return True

        if decision.goal_satisfied:
            self.kernel.task_manager.complete_step(step_description, task.id)
            if plan_step_id:
                self.kernel.task_manager.set_plan_step_status(
                    plan_step_id,
                    PlanStepStatus.COMPLETED,
                    task.id,
                )
            self.kernel.task_manager.add_evidence(
                evidence,
                task.id,
            )
            self.event_bus.emit(
                "task_goal_verified",
                task_id=task.id,
                verification=decision_payload,
            )
            self.kernel.task_manager.complete(
                result={
                    "agent_decision": decision_payload,
                    "tool": result.tool,
                    "output": result.output,
                },
                task_id=task.id,
            )
            return True

        next_action = decision.next_action
        if next_action is None:
            self.kernel.task_manager.add_evidence(
                evidence,
                task.id,
            )
            if decision.needs_observation:
                self.event_bus.emit(
                    "task_agent_observation_required",
                    task_id=task.id,
                    decision=decision_payload,
                )
                self.kernel.task_manager.pause(task.id)
            return True

        self.kernel.task_manager.complete_step(step_description, task.id)
        if plan_step_id:
            self.kernel.task_manager.set_plan_step_status(
                plan_step_id,
                PlanStepStatus.COMPLETED,
                task.id,
            )
            self.kernel.task_manager.refresh_ready_plan_steps(task.id)

        next_step_id = self._next_agent_step_id(task)
        next_step_description = self._describe_command(next_action)
        action = str(next_action.get("action") or "").strip()
        tool_name = str(next_action.get("tool") or "").strip()
        try:
            definition = (
                self.kernel.tool_registry.get(tool_name).definition
                if tool_name
                else None
            )
        except KeyError:
            definition = None

        if definition is None:
            self.kernel.task_manager.fail(
                "Agent selected a capability that could not be resolved.",
                task.id,
            )
            return True

        new_step = PlanStep(
            id=next_step_id,
            description=next_step_description or definition.name,
            status=PlanStepStatus.PENDING,
            depends_on=[plan_step_id] if plan_step_id else [],
            required_capabilities=[definition.name],
            completion_conditions=[
                f"{definition.name} reports success",
            ],
            metadata={
                key: value
                for key, value in next_action.items()
                if key not in {"description"}
                and value is not None
                and value != ""
            },
        )
        new_step.metadata.setdefault("action", action)
        new_step.metadata.setdefault("tool", definition.name)
        new_step.metadata["agent_generated"] = True
        task.plan.add_step(new_step)
        task.pending_steps.append(new_step.description)
        task.plan.status = PlanStatus.ACTIVE
        task.plan.mark_ready_steps()

        next_request = self.build_plan_request(task, new_step)
        if next_request is None:
            self.kernel.task_manager.fail(
                f"Unable to build an agent-selected step '{new_step.id}'.",
                task.id,
            )
            return True

        self.kernel.task_manager.add_evidence(
            evidence,
            task.id,
        )
        print(
            f"[Agent] Next action: {new_step.description} -> "
            f"{next_request.tool}",
            flush=True,
        )
        self.event_bus.emit(
            "tool_request",
            request=next_request,
        )
        return True

    @staticmethod
    def _should_follow_planned_grounded_action(
        task,
        result: ToolResult,
        current_step_id: str | None,
    ) -> bool:
        """Follow a preplanned click immediately after successful vision.locate."""
        if result.tool != "vision.locate" or not result.success:
            return False
        if not current_step_id or task.plan is None:
            return False

        output = result.output if isinstance(result.output, dict) else {}
        located_target = str(
            output.get("target")
            or output.get("element")
            or ""
        ).strip().lower()
        if not located_target:
            return False

        steps = list(task.plan.steps or ())
        for index, candidate in enumerate(steps):
            if candidate.id != current_step_id:
                continue
            if index + 1 >= len(steps):
                return False

            next_step = steps[index + 1]
            metadata = next_step.metadata or {}
            if str(metadata.get("tool") or "").strip() != "computer.click":
                return False

            next_target = str(
                metadata.get("target") or ""
            ).strip().lower()
            dependencies = list(next_step.depends_on or ())
            return (
                current_step_id in dependencies
                and bool(next_target)
                and next_target == located_target
            )

        return False

    _EXPLICIT_VERIFICATION_MARKERS = (
        "verify", "check", "confirm", "make sure", "see if", "ensure",
        "visible", "screenshot", "screen", "look at",
    )

    @classmethod
    def _post_action_skip_reason(
        cls,
        task,
        result: ToolResult,
        current_step_id: str | None,
        *,
        evidence: dict[str, Any],
    ) -> str | None:
        """Return why a post-action LLM decision is unnecessary, if it is.

        A successful, non-visual step whose plan still has work left simply
        continues with the next planned step, and a final step that the
        deterministic verifier already confirmed completes the task. The LLM
        decision stays for visual results, unverified final steps, and goals
        that explicitly ask to check or verify something.
        Set ASTA_AGENT_POST_ACTION=always to restore a decision after every step.
        """
        mode = os.getenv("ASTA_AGENT_POST_ACTION", "auto").strip().lower()
        if mode in {"always", "on", "1", "true"}:
            return None
        if not result.success or str(result.tool or "").startswith("vision."):
            return None
        if not current_step_id or task.plan is None:
            return None

        steps = list(task.plan.steps or ())
        index = next(
            (i for i, step in enumerate(steps) if step.id == current_step_id),
            None,
        )
        if index is None:
            return None

        remaining = [
            step
            for step in steps[index + 1:]
            if getattr(step, "status", None) not in {
                PlanStepStatus.COMPLETED,
                PlanStepStatus.SKIPPED,
            }
        ]
        if remaining:
            return "Step succeeded and the plan has more steps"

        verification = evidence.get("verification")
        verified = (
            isinstance(verification, dict)
            and str(verification.get("status") or "").lower() == "verified"
        )
        goal = " ".join(str(getattr(task, "goal", "") or "").lower().split())
        explicit = any(marker in goal for marker in cls._EXPLICIT_VERIFICATION_MARKERS)
        if verified and not explicit:
            return "Final step verified deterministically"
        if verified and (getattr(task, "metadata", None) or {}).get("intent_type") == IntentType.TASK.value:
            # A project check ("make sure the tests pass") was observed by
            # the command/file verifier; ASTA's report states the evidence.
            return "Project check verified deterministically"
        return None

    @staticmethod
    def _should_follow_planned_visual_verification(task, current_step_id: str | None) -> bool:
        if not current_step_id or task.plan is None:
            return False

        steps = list(task.plan.steps or ())
        for index, candidate in enumerate(steps):
            if candidate.id != current_step_id:
                continue
            if index + 1 >= len(steps):
                return False

            next_step = steps[index + 1]
            metadata = next_step.metadata or {}
            if str(metadata.get("tool") or "").strip() != "vision.inspect":
                return False

            dependencies = list(next_step.depends_on or ())
            return current_step_id in dependencies

        return False

    @staticmethod
    def _next_agent_step_id(task) -> str:
        counter = int(task.metadata.get("agent_step_counter", 0)) + 1
        task.metadata["agent_step_counter"] = counter
        return f"agent-step-{counter}"

    @staticmethod
    def _apply_agent_decision(task, decision: dict[str, Any]) -> None:
        state = task.metadata.get("agent_state")
        if not isinstance(state, dict):
            return

        updates = decision.get("belief_updates") or []
        beliefs = state.setdefault("beliefs", {})
        for update in updates:
            if not isinstance(update, dict):
                continue
            key = str(update.get("key") or "").strip()
            if not key:
                continue
            try:
                confidence = max(
                    0.0,
                    min(1.0, float(update.get("confidence", 0.5))),
                )
            except (TypeError, ValueError):
                confidence = 0.5
            beliefs[key] = {
                "key": key,
                "value": update.get("value"),
                "confidence": confidence,
                "source": str(
                    update.get("source") or "agent_decision"
                ).strip(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        state["current_strategy"] = str(
            decision.get("rationale") or state.get("current_strategy") or ""
        ).strip()
        try:
            state["uncertainty"] = max(
                0.0,
                min(1.0, float(decision.get("uncertainty", state.get("uncertainty", 1.0)))),
            )
        except (TypeError, ValueError):
            pass

        state["last_decision"] = dict(decision)
        history = state.setdefault("decision_history", [])
        history.append(dict(decision))
        if len(history) > 32:
            del history[:-32]

    @staticmethod
    def _record_agent_observation(task, result: ToolResult) -> None:
        state = task.metadata.get("agent_state")
        if not isinstance(state, dict):
            return

        observations = state.setdefault("observations", [])
        observations.append(
            {
                "kind": "tool_result",
                "summary": (
                    f"{result.tool} "
                    f"{'succeeded' if result.success else 'failed'}."
                ),
                "source": result.tool,
                "data": {
                    "success": bool(result.success),
                    "output": result.output,
                    "error": result.error,
                },
            }
        )
        if len(observations) > 32:
            del observations[:-32]

        actions = state.setdefault("actions", [])
        actions.append(
            {
                "action": "execute",
                "tool": result.tool,
                "outcome": "success" if result.success else "failure",
                "error": result.error,
            }
        )
        if len(actions) > 32:
            del actions[:-32]

    def _remember_opened_application(self, result: ToolResult) -> None:
        if not result.success or result.tool not in {
            "system.open_application",
            "system.launch_application",
        }:
            return

        output = result.output
        if not isinstance(output, dict):
            return

        target = str(output.get("target") or "").strip()
        if not target:
            return

        manager = getattr(self.kernel, "application_manager", None)
        remember_opened = getattr(manager, "remember_opened", None)
        resolve = getattr(manager, "resolve", None)
        if not callable(remember_opened) or not callable(resolve):
            return

        try:
            application = resolve(target)
        except Exception:
            return

        try:
            remember_opened(application)
        except Exception:
            return

    def on_capability_setup_completed(self, event):
        if not isinstance(event, dict):
            return

        task_id = str(event.get("task_id") or "").strip()
        task = self.kernel.task_manager.get(task_id) if task_id else None
        if task is None:
            return

        status = str(event.get("status") or "").strip().lower()
        setup_evidence = {
            "type": "capability_setup",
            "capability": event.get("capability"),
            "status": status,
            "summary": event.get("summary"),
            "step_id": event.get("step_id"),
            "requires_user": bool(event.get("requires_user")),
        }
        sandbox = str(event.get("capability") or "").lower() == "sandbox"
        if sandbox:
            details = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            setup_evidence["observed"] = list(details.get("observed") or [])
            setup_evidence["actions"] = list(details.get("actions") or [])
        self.kernel.task_manager.add_evidence(setup_evidence, task.id)
        task.metadata["capability_setup"] = {
            "capability": event.get("capability"),
            "step_id": event.get("step_id"),
            "status": status,
        }
        if sandbox:
            print(
                f"[Tasks] Sandbox repair {status}: observed "
                f"{'; '.join(setup_evidence['observed']) or 'nothing'}; "
                f"did {'; '.join(setup_evidence['actions']) or 'nothing'}.",
                flush=True,
            )
            if status != "completed":
                self.kernel.task_manager.fail(
                    str(event.get("summary") or SANDBOX_SETUP_MESSAGE),
                    task.id,
                )
                return
            # Shown in the chat, not spoken: the result that follows matters.
            self.event_bus.emit("assistant_sentence", text=str(event.get("summary") or ""), spoken="")

        if status != "completed":
            return

        step_id = str(event.get("step_id") or "").strip()
        if task.plan is None or not step_id:
            return

        try:
            step = task.plan.get_step(step_id)
        except KeyError:
            return

        step.status = PlanStepStatus.READY
        task.current_step = None
        task.error = None
        task.plan.status = PlanStatus.ACTIVE

        try:
            self.kernel.task_manager.resume(task.id)
        except ValueError:
            if task.status.value != "active":
                return

        self.kernel.task_manager.refresh_ready_plan_steps(task.id)

        next_step = self._next_ready_plan_step(task)
        if next_step is None:
            return

        next_request = self.build_plan_request(task, next_step)
        if next_request is None:
            self.kernel.task_manager.fail(
                f"Unable to resume task step '{next_step.id}' after capability setup.",
                task.id,
            )
            return

        if sandbox and next_request.tool == "system.run_command":
            # The user already approved exactly this command; it never ran.
            next_request.metadata["retry_after_setup"] = True
        print(
            f"[Tasks] Capability setup complete; resuming {next_step.id} -> "
            f"{next_request.tool}",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=next_request)

    def _start_sandbox_repair(self, task, plan_step_id, result, evidence) -> bool:
        """Diagnose and fix the sandbox environment, then resume the step."""
        if os.getenv("ASTA_SANDBOX_SELF_REPAIR", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        setup_manager = getattr(self.kernel, "capability_setup_manager", None)
        supports = getattr(setup_manager, "supports", None)
        if not plan_step_id or not callable(supports) or not supports("sandbox"):
            return False
        attempts = int(task.metadata.get("sandbox_repair_attempts", 0) or 0)
        if attempts >= 1:
            return False
        task.metadata["sandbox_repair_attempts"] = attempts + 1
        self.kernel.task_manager.set_plan_step_status(
            plan_step_id,
            PlanStepStatus.FAILED,
            task.id,
        )
        evidence["sandbox_repair"] = {"started": True, "error": str(result.error or "")[:300]}
        self.kernel.task_manager.add_evidence(evidence, task.id)
        # Mark the repair as running before pausing, so the pause is not
        # announced as a problem: the repair explains itself.
        task.metadata["capability_setup"] = {
            "capability": "sandbox",
            "step_id": plan_step_id,
            "status": "running",
        }
        self.kernel.task_manager.pause(task.id)
        print(
            "[Tasks] Sandbox isn't ready; diagnosing and repairing it, then "
            f"rerunning {plan_step_id}.",
            flush=True,
        )
        self.event_bus.emit(
            "task_progress",
            task_id=task.id,
            phase="recovery",
            text="The code sandbox isn't ready. I'm checking why and fixing it.",
        )
        started = setup_manager.start(
            task,
            capability="sandbox",
            step_id=plan_step_id,
            context={"error": str(result.error or "")},
        )
        if not started:
            self.kernel.task_manager.fail(SANDBOX_SETUP_MESSAGE, task.id)
        return True

    @staticmethod
    def _setup_provider_for_step(task, plan_step_id, result):
        if task.plan is not None and plan_step_id:
            try:
                step = task.plan.get_step(plan_step_id)
            except KeyError:
                step = None
            if step is not None:
                provider = str(step.metadata.get("provider") or "").strip().lower()
                if provider:
                    return provider

        error = str(result.error or "").lower()
        if result.tool == "media.control" and "spotify" in error:
            return "spotify"
        return ""

    def on_confirmation_response(self, request_id, approved):
        if not isinstance(request_id, str) or not isinstance(approved, bool):
            return

        pending = self.kernel.approval_manager.get(request_id)
        if pending is None:
            return

        task_id = pending.request.metadata.get("task_id")
        task = self.kernel.task_manager.get(task_id) if task_id else None
        if task is None or task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return

        if not approved:
            self.kernel.task_manager.fail(
                "Tool execution rejected by user.",
                task.id,
            )
        else:
            task_step = pending.request.metadata.get("task_step")
            if task_step:
                self.kernel.task_manager.set_step(task_step, task.id)

    @staticmethod
    def _acknowledgment_for_intent(intent: IntentResult) -> str:
        """Create a brief user-facing acknowledgment before execution starts."""
        entities = getattr(intent, "entities", {}) or {}
        commands = entities.get("commands")
        candidates = (
            list(commands)
            if isinstance(commands, list)
            else [entities]
        )

        # Acknowledgment follows the first actionable command so a compound
        # request does not announce a later playback step before the app is
        # even open.
        for command in candidates:
            if not isinstance(command, dict):
                continue
            action = str(command.get("action") or "").strip().lower()
            target = str(command.get("target") or "").strip()

            if action in {"open", "launch", "start"} and target:
                return f"Okay, sir. Opening {target}."

            if action == "media":
                operation = str(command.get("operation") or "").strip().lower()
                query = str(command.get("query") or "").strip()
                requested = str(command.get("requested_provider") or "").strip()
                provider_name = str(command.get("provider") or "").strip().lower()
                if not requested and provider_name in UNSUPPORTED_MUSIC_APPS:
                    requested, command = provider_name, {**command, "provider": "spotify"}
                if operation == "play" and query and requested:
                    target = f"songs by {command['artist']}" if command.get("artist") else query
                    used = str(command.get("provider") or "Spotify")
                    return (
                        f"Okay, sir. I can't use {requested.title()} yet, so I'm playing "
                        f"{target.title()} on {used.title()}."
                    )
                if operation == "play" and command.get("artist"):
                    return f"Okay, sir. Playing songs by {str(command['artist']).title()}."
                if operation == "play" and query:
                    spoken_query = query.title()
                    return f"Okay, sir. Playing {spoken_query}."

        if len(candidates) > 1:
            return "Okay, sir. I’ll handle that."

        return "Okay, sir. I’m on it."

    @staticmethod
    def _progress_for_request(request: ToolRequest) -> str:
        """Convert an executable tool request into a human-readable activity."""
        tool = str(request.tool or "").strip().lower()
        args = request.arguments or {}

        if tool == "system.open_application":
            target = str(args.get("target") or "the application").strip()
            return f"Opening {target}."
        if tool == "system.close_application":
            target = str(args.get("target") or "the application").strip()
            return f"Closing {target}."
        if tool == "media.control":
            operation = str(args.get("operation") or "media").strip().lower()
            query = str(args.get("query") or "").strip()
            if operation == "play" and query:
                return f"Starting playback for {query}."
            return f"Controlling media: {operation}."
        if tool == "vision.inspect":
            return "Checking the screen."
        if tool == "vision.locate":
            target = str(args.get("target") or "the target").strip()
            return f"Locating {target} on the screen."
        if tool == "vision.screenshot":
            return "Taking a look at the screen."
        if tool == "computer.click":
            target = str(args.get("target") or "").strip()
            return (
                f"Clicking {target}."
                if target
                else "Clicking the selected control."
            )
        if tool == "computer.type_text":
            return "Typing into the selected field."
        if tool == "computer.keypress":
            return f"Pressing {args.get('key', 'the key')}."
        if tool == "computer.hotkey":
            keys = args.get("keys") or []
            if isinstance(keys, list):
                shortcut = " + ".join(str(key) for key in keys)
            else:
                shortcut = str(keys)
            return f"Pressing {shortcut or 'a keyboard shortcut'}."
        if tool == "computer.move_mouse":
            return "Moving the pointer to the selected location."
        if tool == "computer.scroll":
            return "Scrolling the current screen."
        if tool == "computer.wait":
            try:
                seconds = float(args.get("seconds", 0.5))
            except (TypeError, ValueError):
                seconds = 0.5
            return (
                f"Waiting {max(0.0, seconds):.1f} seconds "
                "for the interface to settle."
            )
        if tool == "system.run_command":
            return (
                "Running Python in an isolated container. "
                "Only a sanitized, read-only workspace snapshot is mounted; "
                "network access is disabled."
            )

        return f"Working with {request.tool}."

    def _emit_progress(self, phase: str, text: str, *, task_id=None):
        self.event_bus.emit(
            "task_progress",
            task_id=task_id,
            phase=str(phase),
            text=str(text),
        )

    @staticmethod
    def _describe_command(command: Any) -> str:
        if not isinstance(command, dict):
            return str(command)

        action = str(command.get("action") or "command").strip()
        target = str(command.get("target") or "").strip()
        return f"{action} {target}".strip()

    @staticmethod
    def _next_ready_plan_step(task):
        if task.plan is None:
            return None

        for step in task.plan.steps:
            if step.status is PlanStepStatus.READY:
                return step

        return None

    def build_plan_request(self, task, step) -> ToolRequest | None:
        action = str(step.metadata.get("action") or "").strip()
        target = str(step.metadata.get("target") or "").strip()
        if not action:
            return None

        explicit_coordinates = {}
        if action.lower() == "click":
            if (
                step.metadata.get("x") is not None
                and step.metadata.get("y") is not None
            ):
                explicit_coordinates = {
                    "x": step.metadata.get("x"),
                    "y": step.metadata.get("y"),
                }

        grounded = (
            explicit_coordinates
            or self._ground_action_arguments(
                task,
                action=action,
                target=target,
            )
        )
        if grounded is None:
            return None

        raw_entities = {
            "action": action,
            **({"target": target} if target else {}),
            **grounded,
        }
        request_only_metadata = {
            "action",
            "tool",
            "description",
            "verification",
            "expected_returncode",
            "expected_stdout",
            "stdout_match",
            "completion_conditions",
            "sequence_index",
            "depends_on",
            "required_capabilities",
            "agent_generated",
            "user_directed",
            "user_request_prefix",
            "media_query",
            "media_app",
            "completion_message",
        }
        raw_entities.update(
            {
                key: value
                for key, value in step.metadata.items()
                if key not in request_only_metadata
                and value is not None
                and value != ""
            }
        )
        expected_sha256_from_step = raw_entities.pop(
            "expected_sha256_from_step",
            None,
        )
        if expected_sha256_from_step is not None:
            if action.lower() != "write_file":
                print(
                    "[Tasks] A read-result hash can only authorize a file update.",
                    flush=True,
                )
                return None
            digest = self._read_step_digest(
                task,
                source_step_id=expected_sha256_from_step,
                expected_path=raw_entities.get("path"),
            )
            if digest is None:
                print(
                    "[Tasks] The requested prior file-read evidence is missing "
                    "or does not match the update path.",
                    flush=True,
                )
                return None
            if raw_entities.get("expected_sha256") not in (None, "", digest):
                print(
                    "[Tasks] A planned file update supplied conflicting read hashes.",
                    flush=True,
                )
                return None
            raw_entities["expected_sha256"] = digest

        selection_intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=float(
                task.metadata.get("confidence", 0.98)
            ),
            normalized_text=task.goal,
            entities=raw_entities,
            requires_tools=True,
            classifier=str(task.metadata.get("classifier", "rules")),
        )

        try:
            definition = self.tool_request_builder.selector.select(selection_intent)
        except ValueError as exc:
            print(
                f"[Tasks] Could not build next plan step: {exc}",
                flush=True,
            )
            return None

        schema = definition.input_schema if isinstance(definition.input_schema, dict) else {}
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, dict) else {}
        request_entities = {
            "action": action,
            **{
                key: value
                for key, value in raw_entities.items()
                if key != "action" and key in properties
            },
        }
        required = schema.get("required", ())
        if isinstance(required, (list, tuple, set, frozenset)):
            missing = [
                name
                for name in required
                if name != "action" and name not in request_entities
            ]
            if missing:
                print(
                    "[Tasks] Planned tool step omitted required input(s): "
                    + ", ".join(str(name) for name in missing),
                    flush=True,
                )
                return None

        request_intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=selection_intent.confidence,
            normalized_text=selection_intent.normalized_text,
            entities=request_entities,
            requires_tools=True,
            classifier=selection_intent.classifier,
        )
        try:
            request = self.tool_request_builder.build(request_intent)
        except ValueError as exc:
            print(
                f"[Tasks] Could not build next plan step: {exc}",
                flush=True,
            )
            return None
        if request.tool != definition.name:
            print(
                f"[Tasks] Tool selection changed while building step "
                f"('{definition.name}' -> '{request.tool}').",
                flush=True,
            )
            return None

        request.metadata["task_id"] = task.id
        request.metadata["task_step"] = step.description
        request.metadata["plan_step_id"] = step.id
        # Keep the request-level planner marker stable for the existing
        # execution/runtime contract. The originating plan planner (including
        # cognitive_v1) remains available on task.plan.metadata["planner"].
        request.metadata["planner"] = "task_runtime"
        if task.plan is not None and task.plan.metadata.get("planner"):
            request.metadata["plan_planner"] = str(
                task.plan.metadata["planner"]
            )
        if "sequence_index" in step.metadata:
            request.metadata["sequence_index"] = step.metadata["sequence_index"]
        if (
            step.metadata.get("user_directed") is True
            and task.plan is not None
            and (
                task.plan.metadata.get("planner") == "deterministic"
                # Leading app/media commands of a mixed request: planned
                # deterministically from the user's words, then the project
                # part was added by the LLM planner (which cannot set this).
                or step.metadata.get("user_request_prefix") is True
            )
        ):
            request.metadata["user_directed"] = True
        if step.metadata.get("completion_message"):
            request.metadata["completion_message"] = str(
                step.metadata["completion_message"]
            )
        protected = self._unrequested_test_edit(task, request)
        if protected:
            # Making a check pass by rewriting its test is not a fix. The
            # user decides whether a test they didn't mention may change.
            request.metadata["confirm_overwrite"] = protected
            request.metadata["confirm_kind"] = "test_edit"
        return request

    @staticmethod
    def _unrequested_test_edit(task, request: ToolRequest) -> str | None:
        """Path of an existing test file this update would change unasked."""
        if request.tool != "filesystem.write_file" or not request.arguments.get("expected_sha256"):
            return None
        path = str(request.arguments.get("path") or "").replace("\\", "/").strip("/")
        if not path or not is_test_path(path):
            return None
        goal = str((task.metadata.get("intent_entities") or {}).get("goal") or task.goal or "")
        named = {item.lower().strip("/") for item in mentioned_paths(goal)}
        if path.lower() in named or path.rsplit("/", 1)[-1].lower() in named:
            return None
        return path

    @staticmethod
    def _read_step_digest(task, *, source_step_id, expected_path) -> str | None:
        plan = getattr(task, "plan", None)
        if plan is None or not isinstance(source_step_id, str):
            return None
        try:
            source_step = plan.get_step(source_step_id)
        except (KeyError, ValueError):
            return None
        if (
            source_step.status is not PlanStepStatus.COMPLETED
            or source_step.metadata.get("tool") != "filesystem.read_file"
        ):
            return None

        def normalize_path(value):
            return str(value or "").replace("\\", "/").strip("/")

        expected = normalize_path(expected_path)
        if not expected or normalize_path(source_step.metadata.get("path")) != expected:
            return None

        for evidence in reversed(getattr(task, "evidence", ()) or ()):
            if not isinstance(evidence, dict):
                continue
            if (
                evidence.get("plan_step_id") != source_step_id
                or evidence.get("tool") != "filesystem.read_file"
                or evidence.get("success") is not True
            ):
                continue
            output = evidence.get("output")
            if not isinstance(output, dict) or normalize_path(output.get("path")) != expected:
                continue
            digest = output.get("sha256")
            if (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdefABCDEF" for character in digest)
            ):
                return digest.lower()
        return None

    @staticmethod
    def _ground_action_arguments(
        task,
        *,
        action: str,
        target: str,
    ) -> dict[str, Any] | None:
        """Resolve semantic computer actions from the latest visual grounding."""
        normalized_action = str(action or "").strip().lower()
        if normalized_action != "click":
            return {}

        if not target:
            return {}

        agent_state = task.metadata.get("agent_state")
        observations = (
            list(agent_state.get("observations") or ())
            if isinstance(agent_state, dict)
            else []
        )

        evidence = list(getattr(task, "evidence", []) or ())
        for item in evidence:
            if not isinstance(item, dict):
                continue
            if str(item.get("tool") or "").strip().lower() != "vision.locate":
                continue
            observations.append(
                {
                    "source": "vision.locate",
                    "data": {
                        "success": bool(item.get("success")),
                        "output": item.get("output"),
                        "error": item.get("error"),
                    },
                }
            )

        if not observations:
            return None

        wanted = target.strip().lower()

        for observation in reversed(observations):
            if not isinstance(observation, dict):
                continue
            if str(observation.get("source") or "").strip().lower() != "vision.locate":
                continue

            data = observation.get("data")
            if not isinstance(data, dict) or not bool(data.get("success")):
                continue

            output = data.get("output")
            if not isinstance(output, dict):
                continue

            result_target = str(
                output.get("target")
                or output.get("element")
                or ""
            ).strip().lower()
            if result_target != wanted:
                continue

            center = output.get("screen_center") or output.get("center")
            if not isinstance(center, dict):
                continue

            try:
                x = int(round(float(center["x"])))
                y = int(round(float(center["y"])))
            except (KeyError, TypeError, ValueError):
                continue

            return {"x": x, "y": y}

        return None

    @staticmethod
    def _request_plan_step_id(request: ToolRequest, task) -> str | None:
        plan = task.plan
        if plan is None:
            return None

        index = request.metadata.get("sequence_index")
        if isinstance(index, int):
            candidate = f"step-{index + 1}"
            try:
                plan.get_step(candidate)
            except KeyError:
                return None
            return candidate

        for step in plan.steps:
            if step.status in {PlanStepStatus.PENDING, PlanStepStatus.READY}:
                if step.metadata.get("tool") == request.tool:
                    return step.id

        for step in plan.steps:
            if step.status in {PlanStepStatus.PENDING, PlanStepStatus.READY}:
                return step.id

        return None

    @classmethod
    def _request_step(cls, request: ToolRequest) -> str:
        sequence = request.metadata.get("sequence")
        index = request.metadata.get("sequence_index")
        if isinstance(sequence, list) and isinstance(index, int):
            if 0 <= index < len(sequence):
                return cls._describe_command(sequence[index])

        planned_step = request.metadata.get("task_step")
        if isinstance(planned_step, str) and planned_step.strip():
            return planned_step.strip()

        action = request.metadata.get("action") or request.arguments.get("action")
        target = request.arguments.get("target")
        if action:
            return cls._describe_command({"action": action, "target": target})
        return request.tool