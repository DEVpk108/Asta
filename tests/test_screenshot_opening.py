from core import Kernel
from core.contracts import ToolRequest
from core.tools import OpenScreenshotTool, ToolRuntimeModule
from ai.ai_module import AIModule
from ai.runtime_patch import _is_screenshot_open_request, apply_ai_runtime_patch


def test_open_screenshot_phrases_are_deterministic():
    assert _is_screenshot_open_request("Open the screenshot")
    assert _is_screenshot_open_request("Open latest screenshot.")
    assert _is_screenshot_open_request("Show the latest screenshot")
    assert not _is_screenshot_open_request("Open the screenshot application")


def test_open_screenshot_tool_opens_latest_file(tmp_path):
    older = tmp_path / "asta_20260913_170000_000000001.png"
    newer = tmp_path / "asta_20260913_171000_000000002.png"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    older.touch()
    newer.touch()

    opened = []
    tool = OpenScreenshotTool(
        screenshot_dir=tmp_path,
        opener=lambda path: opened.append(path),
    )
    request = ToolRequest(
        tool="vision.open_screenshot",
        arguments={},
        request_id="open-shot-test",
    )

    result = tool.execute(request)

    assert result.success is True
    assert opened == [newer]
    assert result.output["path"] == str(newer.resolve())


def test_open_screenshot_tool_reports_missing_screenshot(tmp_path):
    tool = OpenScreenshotTool(screenshot_dir=tmp_path, opener=lambda path: None)
    request = ToolRequest(
        tool="vision.open_screenshot",
        arguments={},
        request_id="open-shot-empty",
    )

    result = tool.execute(request)

    assert result.success is False
    assert result.error == "No captured screenshots were found."


def test_open_screenshot_request_dispatches_real_tool(tmp_path):
    latest = tmp_path / "asta_20260913_172000_000000001.png"
    latest.write_bytes(b"png")
    opened = []

    kernel = Kernel()
    kernel.register_tool(
        OpenScreenshotTool(
            screenshot_dir=tmp_path,
            opener=lambda path: opened.append(path),
        )
    )

    tool_runtime = ToolRuntimeModule(kernel)
    tool_runtime.initialize()
    apply_ai_runtime_patch()
    ai = AIModule(kernel)
    results = []
    kernel.event_bus.subscribe("tool_result", results.append)

    try:
        ai.on_user_message("Open the screenshot")
    finally:
        tool_runtime.shutdown()

    assert opened == [latest]
    assert results
    assert results[-1].success is True
    assert results[-1].tool == "vision.open_screenshot"
