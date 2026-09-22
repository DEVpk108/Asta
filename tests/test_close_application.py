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
        checks = 0

        def resolve_running_process(self, target):
            assert target == "free download manager"
            self.checks += 1
            if self.checks == 1:
                return RunningProcessRecord(
                    pid=4242,
                    name="fdm",
                    executable_path=r"C:\Program Files\Free Download Manager\fdm.exe",
                    window_title="Free Download Manager",
                )
            raise builtin.ApplicationResolutionError(
                f"No running application matched '{target}'."
            )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
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
    assert calls[0][0] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert calls[0][1]["stdin"] is builtin.subprocess.DEVNULL
    assert calls[0][1]["creationflags"] == getattr(
        builtin.subprocess,
        "CREATE_NO_WINDOW",
        0,
    )


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


def test_close_application_resolves_recent_reference(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")
    calls = []

    class FakeManager:
        checks = 0

        def resolve_reference(self, target):
            assert target == "it"
            return "Spotify"

        def resolve_running_process(self, target):
            assert target == "Spotify"
            self.checks += 1
            if self.checks == 1:
                return RunningProcessRecord(
                    pid=4242,
                    name="Spotify",
                    executable_path=r"C:\Program Files\Spotify\Spotify.exe",
                    window_title="Spotify",
                )
            raise builtin.ApplicationResolutionError(
                f"No running application matched '{target}'."
            )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("it")
    )

    assert result.success is True
    assert result.output["target"] == "it"
    assert result.output["resolved_target"] == "Spotify"
    assert calls[0][0] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert calls[0][1]["stdin"] is builtin.subprocess.DEVNULL
    assert calls[0][1]["creationflags"] == getattr(
        builtin.subprocess,
        "CREATE_NO_WINDOW",
        0,
    )


def test_close_application_compacts_access_denied_errors(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")

    class FakeManager:
        def resolve_reference(self, target):
            return target

        def resolve_running_process(self, target):
            return RunningProcessRecord(
                pid=4242,
                name="Spotify",
            )

    long_error = "\n".join(
        [f"ERROR: The process with PID {pid} could not be terminated.\nReason: Access is denied." for pid in range(1000, 1050)]
    )

    def fake_run(command, **kwargs):
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=long_error,
        )

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("spotify")
    )

    assert result.success is False
    assert result.error.endswith("Access is denied.")
    assert len(result.error) < 160


def test_close_application_fails_when_process_survives_taskkill(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")
    calls = []

    class FakeManager:
        checks = 0

        def resolve_reference(self, target):
            return target

        def resolve_running_process(self, target):
            self.checks += 1
            return RunningProcessRecord(
                pid=4242,
                name="Spotify",
                executable_path=r"C:\Program Files\Spotify\Spotify.exe",
                window_title="Spotify",
            )

    def fake_run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("spotify")
    )

    assert result.success is False
    assert result.output["closed"] is False
    assert "still running" in result.error.lower()
    assert calls == [
        ["taskkill", "/PID", "4242", "/T", "/F"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]

def test_close_application_terminates_all_independent_process_roots(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")
    calls = []

    class FakeManager:
        checks = 0

        def resolve_reference(self, target):
            return target

        def discover_running_process_roots(self, target, limit=64):
            assert target == "sample app"
            self.checks += 1
            if self.checks <= 2:
                return (
                    RunningProcessRecord(
                        pid=100,
                        name="sample",
                        parent_pid=500,
                    ),
                    RunningProcessRecord(
                        pid=200,
                        name="sample",
                        parent_pid=700,
                    ),
                )
            raise builtin.ApplicationResolutionError(
                "No running application matched 'sample app'."
            )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("sample app")
    )

    assert result.success is True
    assert result.output["closed"] is True
    assert result.output["root_pids"] == [100, 200]
    assert result.output["killed_pids"] == [100, 200]
    assert [call[0] for call in calls] == [
        ["taskkill", "/PID", "100", "/T", "/F"],
        ["taskkill", "/PID", "200", "/T", "/F"],
    ]
    assert all(
        call[1]["stdin"] is builtin.subprocess.DEVNULL
        and call[1]["creationflags"] == getattr(
            builtin.subprocess,
            "CREATE_NO_WINDOW",
            0,
        )
        for call in calls
    )


def test_close_application_ignores_root_already_gone_race(monkeypatch):
    monkeypatch.setattr(builtin.platform, "system", lambda: "Windows")

    class FakeManager:
        checks = 0

        def discover_running_process_roots(self, target, limit=64):
            self.checks += 1
            if self.checks == 1:
                return (
                    RunningProcessRecord(
                        pid=4242,
                        name="sample",
                    ),
                )
            raise builtin.ApplicationResolutionError(
                "No running application matched 'sample'."
            )

    def fake_run(command, **kwargs):
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr='ERROR: The process "4242" not found.',
        )

    monkeypatch.setattr(builtin.subprocess, "run", fake_run)

    result = CloseApplicationTool(FakeManager()).execute(
        make_request("sample")
    )

    assert result.success is True
    assert result.output["closed"] is True
