from core import Kernel
from core.context_builder import ContextBuilder
from core.contracts import IntentResult, IntentType
from core.tools import OpenApplicationTool


def test_context_builder_includes_active_task_memory_and_selected_capability():
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    kernel.memory_context = "The user is working on local-first A.S.T.A."
    task = kernel.create_task(
        "open calculator",
        pending_steps=["open calculator"],
        metadata={"source": "test"},
    )

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="open calculator",
        entities={"action": "open", "target": "calculator"},
        requires_tools=True,
    )

    snapshot = ContextBuilder(kernel).build("open calculator", intent)

    assert snapshot.task is not None
    assert snapshot.task["id"] == task.id
    assert snapshot.task["status"] == "active"
    assert snapshot.memory == "The user is working on local-first A.S.T.A."
    assert snapshot.capabilities == (
        {
            "name": "system.open_application",
            "description": "Open a local application, file, URL, or known application alias.",
        },
    )

    prompt = snapshot.to_prompt()
    assert "ACTIVE TASK:" in prompt
    assert "LONG-TERM MEMORY:" in prompt
    assert "AVAILABLE CAPABILITIES FOR THIS TURN:" in prompt
    assert "USER REQUEST:" in prompt
    assert "system.open_application" in prompt


def test_context_builder_does_not_inject_tool_catalog_for_normal_chat():
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    snapshot = ContextBuilder(kernel).build(
        "explain what an event bus is",
        IntentResult(
            intent=IntentType.CONVERSATION,
            confidence=0.98,
            normalized_text="explain what an event bus is",
        ),
    )

    assert snapshot.capabilities == ()
    assert "AVAILABLE CAPABILITIES FOR THIS TURN:" not in snapshot.to_prompt()
