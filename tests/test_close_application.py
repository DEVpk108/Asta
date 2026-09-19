import types

import core.tools.builtin as builtin
from core.applications import RunningProcessRecord
from core.contracts import ToolRequest
from core.tools.builtin import CloseApplicationTool


def make_request(target):
    return ToolRequest(
        tool="system.close_application",
        arguments={"target": target},
        request_id="test-close-app",
    )


def test_close_application_uses_discovered_process_pid(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")
    calls = []

    class FakeManager:
        def resolve_running_process(self, target):
            assert target == "free download manager"
            return RunningProcessRecord(
                pid=4242,
                name="fdm",
                executable_path=r"C:\Program Files\Free Download Manager\fdm.exe",
                window_title="Free Download Manager",
            )

    def fake_run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("free download manager")
    )

    assert result.success is True
    assert result.output == {
        "target": "free download manager",
        "pid": 4242,
        "process": "fdm",
        "closed": True,
    }
    assert calls == [["taskkill", "/PID", "4242", "/T", "/F"]]


def test_close_application_reports_missing_running_application(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")

    class MissingManager:
        def resolve_running_process(self, target):
            raise builtin.ApplicationResolutionError(
                f"No running application matched '{target}'."
            )

    result = CloseApplicationTool(MissingManager()).execute(
        make_request("not running")
    )

    assert result.success is False
    assert result.error == "No running application matched 'not running'."
