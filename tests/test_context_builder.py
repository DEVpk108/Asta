from core import Kernel
from core.context_builder import ContextBuilder
from core.contracts import IntentResult, IntentType
from core.skills import register_builtin_skills
from core.tools import OpenApplicationTool


def test_context_builder_includes_active_task_memory_and_discovered_capability():
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
    assert snapshot.capabilities
    assert snapshot.capabilities[0]["name"] == "system.open_application"
    assert snapshot.capabilities[0]["description"] == (
        "Open a local application, file, URL, or discovered application."
    )
    assert snapshot.capabilities[0]["provider"] == "native"
    assert snapshot.capabilities[0]["loaded"] is True

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


def test_context_builder_uses_workspace_manager_state():
    kernel = Kernel()
    kernel.workspace_manager.update_project(
        name="A.S.T.A.",
        path="C:/Asta",
        repository="https://github.com/DEVpk108/Asta.git",
        branch="feat/asta-hud",
    )
    kernel.workspace_manager.set_active_files(["core/context_builder.py"])

    snapshot = ContextBuilder(kernel).build(
        "what am I working on?",
        IntentResult(
            intent=IntentType.CONVERSATION,
            confidence=0.98,
            normalized_text="what am I working on?",
        ),
    )

    assert snapshot.workspace["project_name"] == "A.S.T.A."
    assert snapshot.workspace["branch"] == "feat/asta-hud"
    assert snapshot.workspace["active_files"] == ["core/context_builder.py"]

    prompt = snapshot.to_prompt()
    assert "WORKSPACE STATE:" in prompt
    assert "project_name: A.S.T.A." in prompt
    assert "branch: feat/asta-hud" in prompt


def test_context_builder_includes_relevant_skill_in_reasoning_context():
    kernel = Kernel()
    register_builtin_skills(kernel.skill_manager)

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="debug the python error",
        entities={"action": "debug", "target": "python error"},
        requires_tools=False,
    )

    snapshot = ContextBuilder(kernel).build(
        "debug the python error",
        intent,
    )

    assert snapshot.skills
    assert snapshot.skills[0]["name"] == "software_debugging"
    assert snapshot.skills[0]["instructions"]

    prompt = snapshot.to_prompt()
    assert "RELEVANT SKILLS FOR THIS TURN:" in prompt
    assert "software_debugging" in prompt
