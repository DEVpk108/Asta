from __future__ import annotations

from typing import Any

from .contracts import (
    IntentResult,
    IntentType,
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
        print("[Tasks] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("tool_request", self.on_tool_request)
        self.event_bus.unsubscribe("tool_result", self.on_tool_result)
        self.event_bus.unsubscribe(
            "tool_confirmation_response",
            self.on_confirmation_response,
        )
        print("[Tasks] Stopped", flush=True)

    def on_user_message(self, text):
        if not text:
            return

        intent = self.kernel.intent_router.analyze(text)
        if intent.intent is not IntentType.COMMAND:
            return

        try:
            plan = self.kernel.planner.plan(
                text,
                intent=intent,
            )
        except PlanningError as exc:
            print(f"[Tasks] Planning failed: {exc}", flush=True)
            return

        pending_steps = [step.description for step in plan.steps]

        task = self.kernel.task_manager.create(
            intent.normalized_text,
            pending_steps=pending_steps,
            plan=plan,
            metadata={
                "intent_type": intent.intent.value,
                "confidence": intent.confidence,
                "classifier": intent.classifier,
            },
        )

        print(
            f"[Tasks] Started task {task.id}: {task.goal}",
            flush=True,
        )

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

        evidence: dict[str, Any] = {
            "type": "tool_result",
            "tool": result.tool,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "request_id": result.metadata.get("request_id"),
        }
        self.kernel.task_manager.add_evidence(evidence, task.id)

        plan_step_id = result.metadata.get("plan_step_id")
        if not result.success:
            if plan_step_id:
                self.kernel.task_manager.set_plan_step_status(
                    plan_step_id,
                    PlanStepStatus.FAILED,
                    task.id,
                )
            self.kernel.task_manager.fail(
                result.error or "Tool execution failed.",
                task.id,
            )
            return

        step = result.metadata.get("task_step") or task.current_step or result.tool
        self.kernel.task_manager.complete_step(step, task.id)

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
        request.metadata["planner"] = "task_runtime"
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

        action = request.metadata.get("action") or request.arguments.get("action")
        target = request.arguments.get("target")
        if action:
            return cls._describe_command({"action": action, "target": target})
        return request.tool
