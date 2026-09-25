from core import Kernel
from core.contracts import ToolRequest
from core.tools import OpenApplicationTool
from ai.ai_module import AIModule


class StubEngine:
    def __init__(self):
        self.system_prompt = ""

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def generate_response(self, *args, **kwargs):
        raise AssertionError("LLM should not be used for this reliability test")


def make_ai():
    kernel = Kernel()
    ai = AIModule(kernel)
    ai.engine = StubEngine()
    return kernel, ai


def test_unknown_name_does_not_call_llm():
    kernel, ai = make_ai()
    responses = []
    kernel.event_bus.subscribe(
        "assistant_response",
        lambda text: responses.append(text),
    )

    ai.on_user_message("What's my name?")

    assert responses == [
        "I don't know your name yet. I don't have that information stored."
    ]


def test_confirmation_is_applied_to_single_pending_request():
    kernel, ai = make_ai()
    confirmations = []
    kernel.event_bus.subscribe(
        "tool_confirmation_response",
        lambda request_id, approved: confirmations.append((request_id, approved)),
    )

    request = ToolRequest(
        tool="system.open_application",
        arguments={"target": "chrome"},
        request_id="approval-test-1",
    )
    kernel.approval_manager.request_approval(request, "confirmation required")

    ai.on_user_message("I confirm")

    assert confirmations == [("approval-test-1", True)]
    assert kernel.approval_manager.contains("approval-test-1")


def test_rejection_is_applied_to_single_pending_request():
    kernel, ai = make_ai()
    confirmations = []
    kernel.event_bus.subscribe(
        "tool_confirmation_response",
        lambda request_id, approved: confirmations.append((request_id, approved)),
    )

    request = ToolRequest(
        tool="system.open_application",
        arguments={"target": "chrome"},
        request_id="approval-test-2",
    )
    kernel.approval_manager.request_approval(request, "confirmation required")

    ai.on_user_message("No")

    assert confirmations == [("approval-test-2", False)]
    assert kernel.approval_manager.contains("approval-test-2")


def test_can_you_open_is_not_treated_as_generic_capability_question():
    kernel, ai = make_ai()
    kernel.register_tool(OpenApplicationTool())
    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    ai.on_user_message("Can you open Chrome?")

    assert len(requests) == 1
    assert requests[0].tool == "system.open_application"
    assert requests[0].arguments == {"target": "chrome"}


def test_grounded_prompt_contains_only_registered_capabilities():
    kernel, ai = make_ai()
    ai._ground_engine_in_capabilities()

    assert "REGISTERED CAPABILITIES:" in ai.engine.system_prompt
    assert "system.open_application" not in ai.engine.system_prompt


def test_tool_failure_is_concise_for_close_application():
    from core.contracts import ToolResult
    result = ToolResult(
        success=False,
        tool="system.close_application",
        output={"target": "Spotify", "closed": False},
        error="Application 'Spotify' was not fully closed. The process is still running.",
    )

    assert AIModule._format_tool_failure(result) == "I couldn't close Spotify."


def test_tool_failure_does_not_speak_raw_diagnostics():
    from core.contracts import ToolResult
    result = ToolResult(
        success=False,
        tool="system.open_application",
        output={"target": "VS Code"},
        error='Traceback (most recent call last): File "main.py", line 10',
    )

    spoken = AIModule._format_tool_failure(result)

    assert spoken == "I couldn't open VS Code."
    assert "Traceback" not in spoken
    assert "main.py" not in spoken



def test_laya_media_recovery_routes_unknown_request_to_media_tool():
    from core.contracts import ActionDecision, ActionType
    from core.media import MediaManager
    from core.tools import MediaControlTool

    class FakeDecisionEngine:
        name = "laya"

        def decide_action(self, text, *, applications=None, media_providers=None):
            return ActionDecision(
                action=ActionType.MEDIA,
                confidence=0.94,
                addressed=0.99,
                arguments={
                    "operation": "play",
                    "query": "hanuman chalisa",
                    "provider": "spotify",
                },
                source="laya",
                model="multilingual",
            )

        def analyze(self, text):
            raise AssertionError("analyze() should not run after media recovery")

        def warmup(self):
            return True

        def shutdown(self):
            return None

    kernel = Kernel()
    kernel.decision_engine = FakeDecisionEngine()
    kernel.register_tool(MediaControlTool(MediaManager(providers=())))
    ai = AIModule(kernel)
    ai.engine = StubEngine()

    requests = []
    kernel.event_bus.subscribe(
        "tool_request",
        lambda request: requests.append(request),
    )

    ai.on_user_message("Play Hanuman Chalisa on Spotify")

    assert len(requests) == 1
    assert requests[0].tool == "media.control"
    assert requests[0].arguments == {
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }



def test_recent_media_plan_recovers_clipped_command_verb():
    from core.contracts import IntentResult, IntentType, Plan, PlanStatus, PlanStep, PlanStepStatus
    from core.task_manager import TaskManager

    kernel = Kernel()
    ai = AIModule(kernel)

    plan = Plan(
        goal="play hanuman chalisa on spotify",
        steps=[
            PlanStep(
                id="step-1",
                description="open Spotify",
                status=PlanStepStatus.COMPLETED,
                metadata={"action": "open", "target": "Spotify", "tool": "system.open_application"},
            ),
            PlanStep(
                id="step-2",
                description="play hanuman chalisa",
                status=PlanStepStatus.READY,
                metadata={
                    "action": "media",
                    "operation": "play",
                    "query": "hanuman chalisa",
                    "provider": "spotify",
                    "tool": "media.control",
                },
            ),
        ],
        status=PlanStatus.ACTIVE,
    )
    kernel.task_manager.create(
        "play hanuman chalisa on spotify",
        pending_steps=["play hanuman chalisa"],
        plan=plan,
    )

    intent = IntentResult(
        intent=IntentType.UNKNOWN,
        confidence=0.20,
        normalized_text="le hanuman chalisa",
    )

    recovered = ai._recover_recent_command("Le Hanuman Chalisa", intent)

    assert recovered is not None
    assert recovered.intent is IntentType.COMMAND
    assert recovered.entities == {
        "action": "media",
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }
    assert recovered.classifier == "task_context"



def test_recent_failed_media_task_recovers_noisy_play_query():
    from core.contracts import (
        IntentResult,
        IntentType,
        Plan,
        PlanStatus,
        PlanStep,
        PlanStepStatus,
    )

    kernel = Kernel()
    ai = AIModule(kernel)

    plan = Plan(
        goal="play hanuman chalisa on spotify",
        steps=[
            PlanStep(
                id="step-1",
                description="open Spotify",
                status=PlanStepStatus.COMPLETED,
                metadata={
                    "action": "open",
                    "target": "Spotify",
                    "tool": "system.open_application",
                },
            ),
            PlanStep(
                id="step-2",
                description="play hanuman chalisa",
                status=PlanStepStatus.READY,
                metadata={
                    "action": "media",
                    "operation": "play",
                    "query": "hanuman chalisa",
                    "provider": "spotify",
                    "tool": "media.control",
                },
            ),
        ],
        status=PlanStatus.ACTIVE,
    )

    task = kernel.task_manager.create(
        "play hanuman chalisa on spotify",
        pending_steps=["play hanuman chalisa"],
        plan=plan,
    )

    task.fail("simulated playback setup failure")

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="play and manjali sir",
        entities={
            "action": "media",
            "operation": "play",
            "query": "and manjali sir",
        },
        requires_tools=True,
        classifier="rules",
    )

    recovered = ai._recover_recent_command(
        "play and manjali sir",
        intent,
    )

    assert recovered is not None
    assert recovered.entities == {
        "action": "media",
        "operation": "play",
        "query": "hanuman chalisa",
        "provider": "spotify",
    }
    assert recovered.classifier == "task_context"
