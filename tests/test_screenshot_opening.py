from pathlib import Path

from core.contracts import ToolRequest
from core.tools import OpenScreenshotTool
from ai.runtime_patch import _is_screenshot_open_request


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
