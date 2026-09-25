from types import SimpleNamespace

from core import Kernel, Planner
from core.autonomy import VerificationEngine, VerificationStatus
from core.contracts import (
    IntentResult,
    IntentType,
    TaskStatus,
    ToolDefinition,
    ToolRequest,
    ToolResult,
)
from core.task_runtime import TaskRuntimeModule
from core.tools import Tool, ToolRuntimeModule


class FakeOpenTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.open",
            description="Open an application for verification tests.",
            input_schema={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["open"]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={"target": request.arguments["target"]},
        )


class FakeMediaPlayTool(Tool):
    @property
    def definition(self):
        return ToolDefinition(
            name="test.media",
            description="Play media for verification tests.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "query": {"type": "string"},
                    "provider": {"type": "string"},
                },
                "required": ["operation"],
            },
            risk_level="low",
            requires_confirmation=False,
            metadata={"actions": ["media"]},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={
                "operation": request.arguments["operation"],
                "query": request.arguments.get("query"),
                "provider": request.arguments.get("provider"),
                "uri": "spotify:track:test",
                "playback_started": True,
            },
        )


class RecordingEngine:
    def generate_response(self, *args, **kwargs):
        raise AssertionError("LLM should not be called for a command intent")


def test_verification_engine_accepts_unconfigured_steps_from_tool_success():
    engine = VerificationEngine()

    task = SimpleNamespace(id="task-1")
    step = SimpleNamespace(id="step-1", metadata={})
    result = SimpleNamespace(success=True, output={"ok": True})

    verification = engine.verify(task, step, result)

    assert verification.status is VerificationStatus.VERIFIED
    assert verification.source == "tool_result"


def test_verification_engine_blocks_unknown_external_verifier():
    engine = VerificationEngine()

    task = SimpleNamespace(id="task-1")
    step = SimpleNamespace(
        id="step-1",
        metadata={"verification": "unknown.capability"},
    )
    result = SimpleNamespace(success=True, output={})

    verification = engine.verify(task, step, result)

    assert verification.status is VerificationStatus.UNKNOWN
    assert verification.source == "guard"


def test_verification_engine_polls_media_observer():
    observations = []

    def observe(provider, query, *, expected_uri=None):
        observations.append((provider, query, expected_uri))
        if len(observations) == 1:
            return {
                "status": "failed",
                "summary": "Playback has not started yet.",
            }
        return {
            "status": "verified",
            "summary": "Observed Hanuman Chalisa playing.",
            "observed": {"uri": expected_uri},
        }

    media = SimpleNamespace(verify_playback=observe)
    engine = VerificationEngine(
        media_manager=media,
        poll_attempts=2,
        poll_delay=0,
    )

    task = SimpleNamespace(id="task-1")
    step = SimpleNamespace(
        id="step-2",
        metadata={
            "verification": "media.playback",
            "provider": "spotify",
            "query": "hanuman chalisa",
        },
    )
    result = SimpleNamespace(
        success=True,
        output={"uri": "spotify:track:test"},
    )

    verification = engine.verify(task, step, result)

    assert verification.status is VerificationStatus.VERIFIED
    assert verification.source == "media_observer"
    assert len(observations) == 2


def test_planner_marks_spotify_play_for_external_verification():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    kernel.register_tool(FakeMediaPlayTool())

    planner = Planner(
        kernel.tool_registry,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )

    intent = IntentResult(
        intent=IntentType.COMMAND,
        confidence=0.98,
        normalized_text="play hanuman chalisa on spotify",
        entities={
            "action": "media",
            "operation": "play",
            "query": "hanuman chalisa",
            "provider": "spotify",
        },
        requires_tools=True,
        classifier="rules",
    )

    plan = planner.plan(
        "play hanuman chalisa on spotify",
        intent=intent,
    )

    assert plan.steps[-1].metadata["verification"] == "media.playback"


def test_runtime_retries_when_external_playback_verification_fails_once():
    kernel = Kernel()
    kernel.register_tool(FakeOpenTool())
    kernel.register_tool(FakeMediaPlayTool())

    verification_calls = 0

    def observe(provider, query, *, expected_uri=None):
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 1:
            return {
                "status": "failed",
                "summary": "Spotify is not currently playing.",
            }
        return {
            "status": "verified",
            "summary": "Observed Spotify playing Hanuman Chalisa.",
            "observed": {
                "uri": expected_uri,
                "track": "Hanuman Chalisa",
                "is_playing": True,
            },
        }

    kernel.media_manager.verify_playback = observe
    kernel.verification_engine = VerificationEngine(
        kernel.media_manager,
        poll_attempts=1,
        poll_delay=0,
    )
    kernel.planner = Planner(
        kernel.tool_registry,
        media_manager=kernel.media_manager,
        application_manager=kernel.application_manager,
    )

    tasks = TaskRuntimeModule(kernel)
    tools = ToolRuntimeModule(kernel)
    ai = __import__("ai.ai_module", fromlist=["AIModule"]).AIModule(kernel)
    ai.engine = RecordingEngine()

    tasks.initialize()
    ai.initialize()
    tools.initialize()

    try:
        kernel.event_bus.emit(
            "user_message",
            "play hanuman chalisa on spotify",
        )

        task = kernel.task_manager.list()[0]
        assert task.status is TaskStatus.COMPLETED
        assert verification_calls == 2
        assert task.metadata["replan_attempts"] == 0

        verification_evidence = [
            item["verification"]
            for item in task.evidence
            if "verification" in item
        ]
        assert verification_evidence[0]["status"] == "failed"
        assert verification_evidence[-1]["status"] == "verified"
    finally:
        tools.shutdown()
        ai.shutdown()
        tasks.shutdown()
