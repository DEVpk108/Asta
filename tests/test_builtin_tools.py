from core.contracts import ToolRequest
from core.tools import (
    AudioControlTool,
    CloseApplicationTool,
    LaunchApplicationTool,
    RunCommandTool,
    ScreenshotTool,
    StartProcessTool,
    StopProcessTool,
)


def request(tool, arguments=None, request_id="test"):
    return ToolRequest(
        tool=tool,
        arguments=arguments or {},
        request_id=request_id,
    )


def test_launch_tool_requires_target():
    result = LaunchApplicationTool().execute(request("system.launch_application"))
    assert not result.success
    assert "target" in result.error


def test_start_process_rejects_non_string_arguments(monkeypatch):
    result = StartProcessTool().execute(
        request("system.start_process", {"target": "python", "arguments": [123]})
    )
    assert not result.success
    assert "list of strings" in result.error


def test_run_command_never_uses_shell(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("core.tools.builtin.shutil.which", lambda _: "tool")
    monkeypatch.setattr("core.tools.builtin.subprocess.run", fake_run)

    result = RunCommandTool().execute(
        request("system.run_command", {"target": "echo", "arguments": ["hello"]})
    )

    assert result.success
    assert calls[0][0] == ["tool", "hello"]
    assert calls[0][1]["shell"] is False if "shell" in calls[0][1] else True


def test_stop_process_rejects_invalid_pid():
    result = StopProcessTool().execute(
        request("system.stop_process", {"target": "not-a-pid"})
    )
    assert not result.success
    assert "process ID" in result.error


def test_close_application_uses_native_process_control(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("core.tools.builtin.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "core.tools.builtin.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)) or Result(),
    )

    result = CloseApplicationTool().execute(
        request("system.close_application", {"target": "calculator"})
    )

    assert result.success
    assert calls[0][0] == ["pkill", "-TERM", "-x", "calculator"]


def test_screenshot_tool_can_use_injected_backend():
    result = ScreenshotTool(capture=lambda: "shot.png").execute(
        request("vision.screenshot")
    )
    assert result.success
    assert result.output == "shot.png"


def test_audio_tool_can_use_injected_backend():
    result = AudioControlTool("mute", handler=lambda: {"muted": True}).execute(
        request("audio.mute")
    )
    assert result.success
    assert result.output["muted"] is True
