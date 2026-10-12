"""Mixed "open/play X, then write and run code" tasks through the real runtimes."""
from __future__ import annotations

from core.background_tasks import BackgroundTaskCoordinator
from core.contracts import IntentResult, IntentType, TaskStatus, ToolDefinition, ToolResult
from core.tools import Tool
from core.tools.approval import ApprovalManager
from tests.test_workspace_project_task import FILE_CONTENT, _build_runtime, _happy_plan, _shutdown

REQUEST = "type baithi hai, then create main.py that prints ASTA_OK and run it"


class FakeTypeTool(Tool):
    def __init__(self):
        self.typed = []

    @property
    def definition(self):
        return ToolDefinition(
            name="computer.type_text",
            description="Type text (test double, same high risk as the real tool).",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            risk_level="high",
            requires_confirmation=False,
            metadata={"actions": ["type_text"], "category": "computer"},
        )

    def execute(self, request):
        self.typed.append(request.arguments.get("text"))
        return ToolResult(success=True, tool="computer.type_text", output={"typed": True})


def _intent(prefix):
    return IntentResult(
        intent=IntentType.TASK,
        confidence=0.9,
        normalized_text=REQUEST,
        entities={
            "goal": "create main.py that prints ASTA_OK and run it",
            "request": REQUEST,
            "capability_scope": [
                "filesystem.list_files",
                "filesystem.read_file",
                "filesystem.write_file",
                "system.run_command",
            ],
            "prefix_commands": prefix,
        },
        requires_tools=True,
        requires_reasoning=True,
        classifier="project_task_rules",
    )


TYPE_STEP = {"action": "type_text", "tool": "computer.type_text", "text": "baithi hai"}


def test_users_own_prefix_typing_runs_without_an_approval(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", _happy_plan())
    typer = FakeTypeTool()
    kernel.register_tool(typer)
    try:
        task = tasks.start_goal(REQUEST, _intent([{**TYPE_STEP, "user_directed": True}]))

        assert typer.typed == ["baithi hai"]
        assert kernel.approval_manager.list_pending() == ()
        assert task.status is TaskStatus.COMPLETED
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
    finally:
        _shutdown(tasks, ai, tools)


def test_model_proposed_typing_still_needs_approval_and_expires_cleanly(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    # The LLM tries to mark its own keyboard step as consented.
    plan = [{**TYPE_STEP, "user_directed": True, "user_request_prefix": True}] + _happy_plan()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", plan)
    kernel.approval_manager = ApprovalManager(ttl_seconds=5)
    monkeypatch.setattr(tools, "_schedule_expiry", lambda request_id: None)
    typer = FakeTypeTool()
    kernel.register_tool(typer)
    intent = _intent([])
    intent.entities.pop("prefix_commands")
    intent.entities["capability_scope"].append("computer.type_text")
    spoken = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    try:
        task = tasks.start_goal(REQUEST, intent)
        pending = kernel.approval_manager.list_pending()
        assert len(pending) == 1 and pending[0].request.tool == "computer.type_text"
        assert typer.typed == []

        monkeypatch.setattr(
            "core.tools.approval.time.monotonic",
            lambda: pending[0].created_at + 6,
        )
        tools._expire_approval(pending[0].request.request_id)

        assert typer.typed == []
        assert task.status is TaskStatus.FAILED
        assert kernel.approval_manager.get(pending[0].request.request_id) is None
        assert any("approval timed out" in str(text) for text in spoken)
        assert not (workspace / "main.py").exists()
    finally:
        _shutdown(tasks, ai, tools)


def test_background_status_reply_is_shown_and_spoken():
    from core import Kernel

    kernel = Kernel()
    coordinator = BackgroundTaskCoordinator(kernel)
    sentences = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: sentences.append(text))
    try:
        coordinator.state = {"job_id": "j", "task_id": "t", "phase": "awaiting_approval",
                             "goal": "g", "message": "Review the requested action"}
        assert coordinator.handle_control("project task status") is True
        assert sentences and "awaiting approval" in sentences[-1]
        assert "approve coding step" in sentences[-1]
    finally:
        coordinator.state = None
        coordinator.shutdown()
        kernel.shutdown()


class FakeInspectTool(Tool):
    def __init__(self, verified):
        self.verified = verified
        self.calls = 0

    @property
    def definition(self):
        return ToolDefinition(
            name="vision.inspect",
            description="Inspect the screen (test double).",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            risk_level="low",
            metadata={"actions": ["inspect"], "category": "vision"},
        )

    def execute(self, request):
        self.calls += 1
        return ToolResult(
            success=True,
            tool="vision.inspect",
            output={"verified": self.verified, "visual_match": self.verified,
                    "summary": "The player bar is not visible.", "confidence": 0.0},
        )


PLAYBACK_CHECK = {
    "action": "inspect",
    "tool": "vision.inspect",
    "prompt": "Verify that 'Baithi Hai' is actually playing in spotify. Inspect only the app.",
    "media_query": "Baithi Hai",
    "media_app": "spotify",
}


def test_unconfirmed_music_check_does_not_drop_the_project_part(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", _happy_plan())
    inspector = FakeInspectTool(verified=False)
    kernel.register_tool(inspector)
    spoken = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    try:
        task = tasks.start_goal(REQUEST, _intent([dict(PLAYBACK_CHECK)]))

        assert inspector.calls == 1
        assert task.status is TaskStatus.COMPLETED
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
        assert any(
            "couldn't confirm that Baithi Hai is playing in spotify" in str(text)
            and "ASTA_OK" in str(text)
            for text in spoken
        )
    finally:
        _shutdown(tasks, ai, tools)


def test_plain_music_command_still_fails_when_playback_is_not_confirmed(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTA_AGENT_PLANNING", "never")  # keep the rules plan
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", _happy_plan())
    kernel.register_tool(FakeInspectTool(verified=False))
    try:
        intent = IntentResult(
            intent=IntentType.COMMAND, confidence=0.98, normalized_text="check playback",
            entities={"commands": [dict(PLAYBACK_CHECK)]}, requires_tools=True, classifier="rules",
        )
        task = tasks.start_plan("check playback", intent)
        assert [step.metadata["tool"] for step in task.plan.steps] == ["vision.inspect"]
        tasks.event_bus.emit("tool_request", request=tasks.build_plan_request(task, tasks._next_ready_plan_step(task)))

        assert task.status is TaskStatus.FAILED
    finally:
        _shutdown(tasks, ai, tools)


def test_inspect_trusts_the_os_media_session_before_vision(tmp_path):
    from core.contracts import ToolRequest
    from core.tools.vision_inspect import VisionInspectTool

    class NoVision:
        def inspect(self, *args, **kwargs):
            raise AssertionError("vision must not run when the media session confirms playback")

    seen = []

    def reader(operation, app):
        seen.append(app)
        return {"ok": True, "app": "Spotify.exe", "title": "Baithi Hai",
                "artist": "Amit Trivedi", "status": "Playing"}

    tool = VisionInspectTool(NoVision(), capture=lambda: tmp_path / "s.png", media_session_reader=reader)
    result = tool.execute(ToolRequest(tool="vision.inspect", arguments={"prompt": PLAYBACK_CHECK["prompt"]}, request_id="inspect-test"))

    assert seen == ["spotify"]
    assert result.success and result.output["verified"] is True
    assert result.output["track"] == "Baithi Hai by Amit Trivedi"


def test_inspect_falls_back_to_vision_when_another_track_is_playing(tmp_path, monkeypatch):
    from core.contracts import ToolRequest
    from core.tools.vision_inspect import VisionInspectTool

    monkeypatch.setattr("core.tools.vision_inspect.time.sleep", lambda seconds: None)
    calls = []

    class Vision:
        def inspect(self, *args, **kwargs):
            calls.append(args)
            return {"json": {"visual_match": False, "confidence": 0.0, "summary": "not visible"}}

    def reader(operation, app):
        return {"ok": True, "title": "Some Other Song", "status": "Playing"}

    (tmp_path / "s.png").write_bytes(b"png")
    tool = VisionInspectTool(Vision(), capture=lambda: tmp_path / "s.png", media_session_reader=reader)
    result = tool.execute(ToolRequest(tool="vision.inspect", arguments={"prompt": PLAYBACK_CHECK["prompt"]}, request_id="inspect-test"))

    assert calls
    assert result.output["verified"] is False


class FakeUIPlayTool(Tool):
    def __init__(self, playing):
        self.playing = playing
        self.calls = 0

    @property
    def definition(self):
        return ToolDefinition(
            name="media.ui_play",
            description="Press play in a music app (test double).",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            risk_level="medium",
            metadata={"actions": ["ui_play"], "category": "media"},
        )

    def execute(self, request):
        self.calls += 1
        output = {"query": "Baithi Hai", "application": "Spotify"}
        if self.playing:
            output.update(verified=True, message="Playing Baithi Hai by Neeru Soni.")
            return ToolResult(success=True, tool="media.ui_play", output=output)
        output["verified"] = False
        return ToolResult(
            success=False,
            tool="media.ui_play",
            output=output,
            error="Pressed play for 'Baithi Hai' in Spotify, but playback did not start.",
        )


UI_PLAY_STEP = {
    "action": "ui_play",
    "tool": "media.ui_play",
    "query": "Baithi Hai",
    "application": "Spotify",
    "process": "Spotify",
}


def test_failed_music_step_still_finishes_the_project_part(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", _happy_plan())
    player = FakeUIPlayTool(playing=False)
    kernel.register_tool(player)
    spoken = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    try:
        task = tasks.start_goal(REQUEST, _intent([dict(UI_PLAY_STEP)]))

        assert player.calls == 1  # no recovery retries / replans of the music part
        assert task.status is TaskStatus.COMPLETED
        assert (workspace / "main.py").read_text(encoding="utf-8") == FILE_CONTENT
        assert any(
            "couldn't confirm that Baithi Hai is playing in Spotify" in str(text)
            and "ASTA_OK" in str(text)
            for text in spoken
        )
    finally:
        _shutdown(tasks, ai, tools)


def test_confirmed_music_step_is_reported_with_the_project_result(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel, tasks, ai, tools = _build_runtime(workspace, tmp_path / "authority.json", _happy_plan())
    kernel.register_tool(FakeUIPlayTool(playing=True))
    spoken = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: spoken.append(text))
    try:
        task = tasks.start_goal(REQUEST, _intent([dict(UI_PLAY_STEP)]))

        assert task.status is TaskStatus.COMPLETED
        assert any(
            str(text).startswith("Playing Baithi Hai by Neeru Soni.") and "ASTA_OK" in str(text)
            for text in spoken
        )
    finally:
        _shutdown(tasks, ai, tools)


def test_music_can_be_paused_while_a_project_task_runs():
    from types import SimpleNamespace

    from ai.ai_module import AIModule
    from core import Kernel
    from core.contracts import ToolDefinition as _Def

    class FakeMediaControl(Tool):
        def __init__(self):
            self.operations = []

        @property
        def definition(self):
            return _Def(name="media.control", description="media", input_schema={"type": "object"},
                        risk_level="low", metadata={"actions": ["media"], "category": "media"})

        def execute(self, request):
            self.operations.append(request.arguments["operation"])
            return ToolResult(success=True, tool="media.control", output={"message": "Paused."})

    kernel = Kernel()
    media = FakeMediaControl()
    kernel.register_tool(media)
    kernel.background_tasks = SimpleNamespace(busy=True, handle_control=lambda text: False)
    ai = AIModule(kernel)
    said = []
    ai._emit_assistant_text = said.append
    ai.on_user_message("Pause the song")
    ai.on_user_message("play Baithi Hai")  # a new song still waits for the task

    assert media.operations == ["pause"]
    assert said[0] == "Paused."
    assert "project task is still running" in said[1]


def test_a_paused_project_task_says_why():
    from core import Kernel

    kernel = Kernel()
    coordinator = BackgroundTaskCoordinator(kernel)
    sentences = []
    kernel.event_bus.subscribe("assistant_sentence", lambda text=None, **kw: sentences.append(text))
    try:
        coordinator.state = {"job_id": "j", "task_id": "t", "phase": "working", "goal": "g", "message": ""}
        coordinator.on_terminal(task={
            "id": "t", "status": "paused", "metadata": {},
            "evidence": [{"type": "tool_result", "tool": "filesystem.write_file", "success": False,
                          "error": "The file changed since it was read; read it again before updating."}],
        })
        assert sentences and sentences[-1].startswith("I paused the project task because of this: The file changed")
        assert "cancel task" in sentences[-1]
    finally:
        coordinator.state = None
        coordinator.shutdown()
        kernel.shutdown()
