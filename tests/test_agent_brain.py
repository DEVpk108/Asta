from core import Kernel
from core.agent import AgentBrain
from core.contracts import IntentResult, IntentType, ToolDefinition
from core.tools import Tool


class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open a target for tests.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request):
        raise AssertionError("agent brain tests do not execute tools")


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def generate_response(self, text, **kwargs):
        self.prompts.append(text)
        return self.response


def _intent():
    return IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="open calculator",
        entities={"action": "open", "target": "calculator"},
        requires_tools=True,
        classifier="rules",
    )


def test_agent_brain_builds_structured_plan_from_local_model():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    provider = FakeProvider(
        """{
            "goal_summary": "Open the calculator application.",
            "success_conditions": ["Calculator is open."],
            "rationale": "The requested goal requires opening the target application.",
            "uncertainty": 0.1,
            "steps": [
                {"action": "open", "target": "calculator"}
            ]
        }"""
    )
    brain = AgentBrain(
        kernel,
        provider=provider,
        enabled=True,
    )

    proposal = brain.plan(
        "open calculator",
        intent=_intent(),
    )

    assert proposal.goal_summary == "Open the calculator application."
    assert proposal.success_conditions == ("Calculator is open.",)
    assert proposal.steps == ({"action": "open", "target": "calculator"},)
    assert proposal.uncertainty == 0.1
    assert provider.prompts


def test_agent_brain_rejects_unsupported_actions():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    provider = FakeProvider(
        """{
            "goal_summary": "Do something unsafe.",
            "success_conditions": ["Done."],
            "rationale": "Test.",
            "uncertainty": 0.5,
            "steps": [
                {"action": "invented_action", "target": "calculator"}
            ]
        }"""
    )
    brain = AgentBrain(
        kernel,
        provider=provider,
        enabled=True,
    )

    try:
        brain.plan("do something", intent=_intent())
    except RuntimeError as exc:
        assert "not executable" in str(exc)
    else:
        raise AssertionError("Expected unsupported action rejection")


class FakeNamedOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="system.open_application",
            description="Open an application.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request):
        raise AssertionError("agent brain tests do not execute tools")


def test_agent_brain_normalizes_tool_name_used_as_action():
    kernel = Kernel()
    kernel.register_tool(FakeNamedOpenTool())
    provider = FakeProvider(
        """{
            "goal_summary": "Open the calculator application.",
            "success_conditions": ["Calculator is open."],
            "rationale": "Use the available application opener.",
            "uncertainty": 0.2,
            "steps": [
                {
                    "action": "system.open_application",
                    "target": "calculator"
                }
            ]
        }"""
    )
    brain = AgentBrain(
        kernel,
        provider=provider,
        enabled=True,
    )

    proposal = brain.plan(
        "open calculator",
        intent=_intent(),
    )

    assert proposal.steps == (
        {
            "action": "open",
            "tool": "system.open_application",
            "target": "calculator",
        },
    )


def test_agent_brain_builds_structured_post_action_decision():
    kernel = Kernel()
    kernel.register_tool(FakeNamedOpenTool())

    provider = FakeProvider(
        """{
            "goal_satisfied": false,
            "needs_observation": true,
            "needs_user": false,
            "rationale": "Opening the application does not prove it is visible, so capture a screenshot.",
            "confidence": 0.78,
            "uncertainty": 0.22,
            "next_action": {
                "action": "vision.screenshot",
                "target": ""
            },
            "belief_updates": [
                {
                    "key": "calculator_launch_succeeded",
                    "value": true,
                    "confidence": 0.88,
                    "source": "tool"
                }
            ]
        }"""
    )
    # Replace the registered capability with a screenshot tool for validation.
    kernel.tool_registry.unregister("system.open_application")
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.screenshot",
                description="Capture a screenshot.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                risk_level="low",
                requires_confirmation=False,
                metadata={"actions": ["screenshot"], "category": "vision"},
            )
        )
    )
    brain = AgentBrain(kernel, provider=provider, enabled=True)

    decision = brain.decide(
        {
            "goal": "open calculator and verify that it is open",
            "success_conditions": ["Calculator is visible."],
            "latest_result": {
                "tool": "system.open_application",
                "success": True,
                "output": {"target": "calculator"},
            },
        }
    )

    assert decision.goal_satisfied is False
    assert decision.needs_observation is True
    assert decision.next_action == {
        "action": "screenshot",
        "tool": "vision.screenshot",
        "target": "",
    }
    assert decision.belief_updates[0]["key"] == "calculator_launch_succeeded"


class ToolDefinitionBackedTool(Tool):
    def __init__(self, definition):
        self._definition = definition

    @property
    def definition(self):
        return self._definition

    def execute(self, request):
        raise AssertionError("decision tests must not execute tools")


def test_agent_brain_rejects_action_without_safe_next_step_or_completion():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    provider = FakeProvider(
        """{
            "goal_satisfied": false,
            "needs_observation": false,
            "needs_user": false,
            "rationale": "I need more evidence.",
            "confidence": 0.4,
            "uncertainty": 0.6
        }"""
    )
    brain = AgentBrain(kernel, provider=provider, enabled=True)

    try:
        brain.decide({"goal": "open calculator"})
    except RuntimeError as exc:
        assert "neither completion nor a next action" in str(exc)
    else:
        raise AssertionError("Expected invalid decision rejection")


def test_agent_brain_blocks_visual_completion_without_visual_evidence():
    kernel = Kernel()
    kernel.register_tool(FakeNamedOpenTool())
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.inspect",
                description="Capture and semantically inspect the current screen.",
                input_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                risk_level="low",
                requires_confirmation=False,
                metadata={
                    "actions": ["inspect", "visual_verify"],
                    "category": "vision",
                },
            )
        )
    )
    provider = FakeProvider(
        """{
            "goal_satisfied": true,
            "needs_observation": false,
            "needs_user": false,
            "rationale": "The application opened successfully, so the goal is complete.",
            "confidence": 1.0,
            "uncertainty": 0.0
        }"""
    )
    brain = AgentBrain(kernel, provider=provider, enabled=True)

    decision = brain.decide(
        {
            "goal": "open calculator and verify that it is open",
            "success_conditions": ["Calculator application is open and visible."],
            "observations": [
                {
                    "kind": "tool_result",
                    "source": "system.open_application",
                    "data": {
                        "success": True,
                        "output": {"target": "calculator"},
                    },
                }
            ],
            "latest_result": {
                "tool": "system.open_application",
                "success": True,
                "output": {"target": "calculator"},
            },
        }
    )

    assert decision.goal_satisfied is False
    assert decision.needs_observation is True
    assert decision.next_action["action"] == "inspect"
    assert decision.next_action["tool"] == "vision.inspect"
    assert "Calculator application is open and visible" in decision.next_action["prompt"]


def test_agent_brain_falls_back_to_screenshot_without_semantic_inspector():
    kernel = Kernel()
    kernel.register_tool(FakeNamedOpenTool())
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.screenshot",
                description="Capture a screenshot.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                risk_level="low",
                requires_confirmation=False,
                metadata={"actions": ["screenshot"], "category": "vision"},
            )
        )
    )
    provider = FakeProvider(
        """{
            "goal_satisfied": true,
            "needs_observation": false,
            "needs_user": false,
            "rationale": "The application opened successfully.",
            "confidence": 1.0,
            "uncertainty": 0.0
        }"""
    )
    brain = AgentBrain(kernel, provider=provider, enabled=True)

    decision = brain.decide(
        {
            "goal": "open calculator and verify that it is open",
            "success_conditions": ["Calculator application is open and visible."],
            "latest_result": {
                "tool": "system.open_application",
                "success": True,
            },
        }
    )

    assert decision.next_action == {
        "action": "screenshot",
        "tool": "vision.screenshot",
    }


def test_agent_brain_allows_visual_completion_with_explicit_verified_evidence():
    kernel = Kernel()
    kernel.register_tool(FakeNamedOpenTool())
    provider = FakeProvider(
        """{
            "goal_satisfied": true,
            "needs_observation": false,
            "needs_user": false,
            "rationale": "The screenshot evidence confirms the calculator window is visible.",
            "confidence": 0.98,
            "uncertainty": 0.02
        }"""
    )
    brain = AgentBrain(kernel, provider=provider, enabled=True)

    decision = brain.decide(
        {
            "goal": "open calculator and verify that it is open",
            "success_conditions": ["Calculator application is open and visible."],
            "observations": [
                {
                    "kind": "visual_verification",
                    "source": "vision.bonsai",
                    "data": {
                        "output": {
                            "visible": True,
                            "verified": True,
                        }
                    },
                }
            ],
            "latest_result": {
                "tool": "vision.inspect",
                "success": True,
                "output": {
                    "visible": True,
                    "verified": True,
                },
            },
        }
    )

    assert decision.goal_satisfied is True


def test_agent_brain_preserves_zero_uncertainty():
    proposal = AgentBrain._parse_response(
        """{
            "goal_summary": "Open calculator.",
            "success_conditions": ["Calculator is open."],
            "rationale": "The request is deterministic.",
            "uncertainty": 0,
            "steps": [{"action": "open", "target": "calculator"}]
        }"""
    )
    assert proposal.uncertainty == 0.0


def test_agent_brain_preserves_zero_decision_uncertainty():
    decision = AgentBrain._parse_decision(
        """{
            "goal_satisfied": true,
            "needs_observation": false,
            "needs_user": false,
            "rationale": "Explicit verification evidence is present.",
            "confidence": 1,
            "uncertainty": 0
        }"""
    )
    assert decision.confidence == 1.0
    assert decision.uncertainty == 0.0

def test_agent_brain_compacts_repeated_decision_state():
    state = {
        "goal": "verify calculator",
        "success_conditions": ["Calculator is visible."],
        "beliefs": {"calculator_open": {"confidence": 0.9}},
        "current_strategy": "Inspect the screen.",
        "uncertainty": 0.1,
        "current_step": {"id": "step-2", "description": "inspect"},
        "observations": [{"data": {"output": {"visible": False}}}] * 20,
        "actions": [{"action": "execute", "tool": "vision.inspect"}] * 20,
        "last_decision": {"goal_satisfied": False},
        "latest_result": {"tool": "vision.screenshot", "success": True},
    }

    compact = AgentBrain._compact_task_state(state)

    assert compact["goal"] == "verify calculator"
    assert compact["latest_result"]["tool"] == "vision.screenshot"
    assert compact["latest_observation"]["data"]["output"]["visible"] is False
    assert "actions" not in compact
    assert "observations" not in compact


def test_agent_brain_compacts_capability_metadata():
    kernel = Kernel()
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.inspect",
                description="Inspect the current screen.",
                input_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                metadata={
                    "actions": ["inspect", "visual_verify"],
                    "category": "vision",
                },
            )
        )
    )

    capabilities = AgentBrain._compact_capabilities(kernel.tool_registry)

    assert capabilities == [{
        "name": "vision.inspect",
        "description": "Inspect the current screen.",
        "actions": ["inspect", "visual_verify"],
        "category": "vision",
        "required_inputs": ["prompt"],
    }]

def test_agent_brain_repairs_truncated_plan():
    response = """{
      "goal_summary": "Open calculator and verify it is visible.",
      "success_conditions": ["Calculator window is visible."],
      "rationale": "Open it, then verify visually.",
      "uncertainty": 0.1,
      "steps": [
        {"action": "open", "tool": "system.open_application", "target": "calculator"},
        {"action": "inspect", "tool": "vision.inspect", "prompt": "Is calculator visible?"
"""
    proposal = AgentBrain._parse_response(response)

    assert proposal.steps == (
        {
            "action": "open",
            "tool": "system.open_application",
            "target": "calculator",
        },
    )
    assert proposal.success_conditions == ("Calculator window is visible.",)


def test_agent_brain_normalizes_visual_open_screenshot_decision():
    kernel = Kernel()
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.open_screenshot",
                description="Open a screenshot.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                metadata={"actions": ["open"], "category": "vision"},
            )
        )
    )
    kernel.register_tool(
        ToolDefinitionBackedTool(
            ToolDefinition(
                name="vision.inspect",
                description="Inspect the current screen.",
                input_schema={
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
                metadata={"actions": ["inspect", "visual_verify"], "category": "vision"},
            )
        )
    )

    decision = AgentDecision(
        goal_satisfied=False,
        needs_observation=True,
        rationale="Use the screenshot.",
        confidence=0.8,
        next_action={"action": "open", "tool": "vision.open_screenshot", "path": "x.png"},
        uncertainty=0.2,
    )
    normalized = AgentBrain(kernel, enabled=True)._normalize_visual_next_action(
        decision,
        {
            "goal": "verify calculator",
            "success_conditions": ["Calculator window is visible."],
        },
    )

    assert normalized.next_action["action"] == "inspect"
    assert normalized.next_action["tool"] == "vision.inspect"
    assert "Calculator window is visible" in normalized.next_action["prompt"]

