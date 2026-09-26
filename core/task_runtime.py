from __future__ import annotations

from typing import Any

from .contracts import (
    IntentResult,
    IntentType,
    PlanStatus,
    PlanStepStatus,
    TaskStatus,
    ToolRequest,
    ToolResult,
)
from .tools.request_builder import ToolRequestBuilder
from .planner import PlanningError
from .module import Module


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
        kernel.task_runtime = self

    def initialize(self):
        self.event_bus.subscribe("user_message", self.on_user_message)
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
        print("[Tasks] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
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
        print("[Tasks] Stopped", flush=True)

    def on_user_message(self, text):
        if not text:
            return

        intent = self.kernel.intent_router.analyze(text)
        if intent.intent is not IntentType.COMMAND:
            return

        self.start_plan(
            intent.normalized_text,
            intent,
        )

    def start_plan(self, goal: str, intent: IntentResult):
        """Create an executable task plan from any structured command intent."""
        try:
            plan = self.kernel.planner.plan(
                goal,
                intent=intent,
            )
        except PlanningError as exc:
            print(f"[Tasks] Planning failed: {exc}", flush=True)
            return None

        pending_steps = [step.description for step in plan.steps]

        metadata = {
            "intent_type": intent.intent.value,
            "confidence": intent.confidence,
            "classifier": intent.classifier,
            "intent_entities": dict(intent.entities),
            "replan_attempts": 0,
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
                "current_strategy": str(
                    plan.metadata.get("agent_rationale") or ""
                ),
                "uncertainty": float(
                    plan.metadata.get("agent_uncertainty", 0.5) or 0.5
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
        return task

    def on_tool_request(self, request):
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

    def on_tool_result(self, result):
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
        if not result.success:
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

                try:
                    new_plan = self.kernel.planner.replan(
                        task,
                        diagnosis,
                        replan_decision.strategy,
                    )
                except PlanningError as exc:
                    evidence["replan_error"] = str(exc)
                    self.kernel.task_manager.add_evidence(evidence, task.id)
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
                self.kernel.task_manager.add_evidence(evidence, task.id)
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
            self.kernel.task_manager.complete(
                result={
                    "tool": result.tool,
                    "output": result.output,
                },
                task_id=task.id,
            )
            return

        next_step = self._next_ready_plan_step(refreshed)
        if next_step is None:
            return

        next_request = self.build_plan_request(refreshed, next_step)
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
        self.kernel.task_manager.add_evidence(setup_evidence, task.id)
        task.metadata["capability_setup"] = {
            "capability": event.get("capability"),
            "step_id": event.get("step_id"),
            "status": status,
        }

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

        print(
            f"[Tasks] Capability setup complete; resuming {next_step.id} -> "
            f"{next_request.tool}",
            flush=True,
        )
        self.event_bus.emit("tool_request", request=next_request)

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

        intent = IntentResult(
            intent=IntentType.COMMAND,
            confidence=float(
                task.metadata.get("confidence", 0.98)
            ),
            normalized_text=task.goal,
            entities={
                "action": action,
                **({"target": target} if target else {}),
                **{
                    key: value
                    for key, value in step.metadata.items()
                    if key in {"operation", "query", "provider"}
                    and value not in {None, ""}
                },
            },
            requires_tools=True,
            classifier=str(task.metadata.get("classifier", "rules")),
        )

        try:
            request = self.tool_request_builder.build(intent)
        except ValueError as exc:
            print(
                f"[Tasks] Could not build next plan step: {exc}",
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
        return request

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