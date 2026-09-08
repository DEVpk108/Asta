from core import Kernel
from core.contracts import ToolRequest
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
    assert not kernel.approval_manager.contains("approval-test-1")


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
    assert not kernel.approval_manager.contains("approval-test-2")


def test_can_you_open_is_not_treated_as_generic_capability_question():
    kernel, ai = make_ai()
    kernel.event_bus.subscribe("tool_request", lambda request: None)

    ai.on_user_message("Can you open Chrome?")


def test_grounded_prompt_contains_only_registered_capabilities():
    kernel, ai = make_ai()
    ai._ground_engine_in_capabilities()

    assert "REGISTERED CAPABILITIES:" in ai.engine.system_prompt
    assert "system.open_application" not in ai.engine.system_prompt
