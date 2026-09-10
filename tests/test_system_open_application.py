import core.tools.system as system
from core.contracts import ToolRequest
from core.tools.system import OpenApplicationTool


def make_request(target):
    return ToolRequest(
        tool="system.open_application",
        arguments={"target": target},
        request_id="test-open-app",
    )


def test_windows_aliases_resolve(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")

    assert OpenApplicationTool.resolve_target("Chrome") == "Chrome"
    assert OpenApplicationTool.resolve_target("calculator") == "calc.exe"
    assert OpenApplicationTool.resolve_target("camera") == "microsoft.windows.camera:"
    assert OpenApplicationTool.resolve_target("spotify") == "spotify:"


def test_windows_path_and_explicit_uri_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(system.os, "name", "nt")
    path = tmp_path / "demo.exe"
    path.write_text("placeholder")

    assert OpenApplicationTool.resolve_target(str(path)) == str(path)
    assert OpenApplicationTool.resolve_target("https://example.com") == "https://example.com"


def test_execute_reports_original_and_resolved_targets(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    opened = []

    monkeypatch.setattr(
        OpenApplicationTool,
        "_open",
        staticmethod(lambda target: opened.append(target)),
    )

    result = OpenApplicationTool().execute(make_request("spotify"))

    assert result.success is True
    assert result.output["target"] == "spotify"
    assert result.output["resolved_target"] == "spotify:"
    assert opened == ["spotify:"]


def test_unknown_windows_application_returns_failure(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    monkeypatch.setattr(system.shutil, "which", lambda target: None)
    monkeypatch.setattr(system.os.path, "exists", lambda target: False)

    result = OpenApplicationTool().execute(make_request("definitely-not-installed"))

    assert result.success is False
    assert "not found" in result.error.lower()
