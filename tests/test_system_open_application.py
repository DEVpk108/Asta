import core.tools.system as system
from core.applications import ApplicationRecord
from core.contracts import ToolRequest
from core.tools.system import OpenApplicationTool


def make_request(target):
    return ToolRequest(
        tool="system.open_application",
        arguments={"target": target},
        request_id="test-open-app",
    )


def test_windows_application_resolves_through_discovery(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")

    record = ApplicationRecord(
        name="WhatsApp",
        launch_target=r"shell:AppsFolder\WhatsApp.App",
        provider="windows.start_apps",
        app_id="WhatsApp.App",
    )

    class FakeManager:
        def resolve(self, target):
            assert target == "whatsapp"
            return record

    tool = OpenApplicationTool(FakeManager())

    assert tool.resolve_target("whatsapp") == record.launch_target


def test_windows_paths_and_uris_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(system.os, "name", "nt")
    tool = OpenApplicationTool(FakeManager())

    path = tmp_path / "demo.exe"
    path.write_text("placeholder")

    assert tool.resolve_target(str(path)) == str(path)
    assert tool.resolve_target("https://example.com") == "https://example.com"


def test_windows_path_resolution_uses_path_before_discovery(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    monkeypatch.setattr(system.shutil, "which", lambda target: r"C:\Tools\editor.exe")

    class FailingManager:
        def resolve(self, target):
            raise AssertionError("discovery should not run for PATH executables")

    tool = OpenApplicationTool(FailingManager())
    assert tool.resolve_target("editor") == r"C:\Tools\editor.exe"


def test_execute_reports_original_and_resolved_targets(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    opened = []

    class FakeManager:
        def resolve(self, target):
            return ApplicationRecord(
                name=target,
                launch_target=r"shell:AppsFolder\Demo.App",
                provider="windows.start_apps",
            )

    monkeypatch.setattr(
        OpenApplicationTool,
        "_open",
        staticmethod(lambda target: opened.append(target)),
    )

    result = OpenApplicationTool(FakeManager()).execute(make_request("demo"))

    assert result.success is True
    assert result.output["target"] == "demo"
    assert result.output["resolved_target"] == r"shell:AppsFolder\Demo.App"
    assert opened == [r"shell:AppsFolder\Demo.App"]
    assert result.metadata["request_id"] == "test-open-app"


def test_windows_store_target_falls_back_to_explorer(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    monkeypatch.setattr(system.shutil, "which", lambda target: r"C:\Windows\explorer.exe")
    calls = []

    def fail_startfile(target):
        raise OSError("startfile failed")

    class FakePopen:
        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(system.os, "startfile", fail_startfile, raising=False)
    monkeypatch.setattr(system.subprocess, "Popen", FakePopen)

    OpenApplicationTool._open(r"shell:AppsFolder\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App")

    assert calls
    assert calls[0][0] == [
        r"C:\Windows\explorer.exe",
        r"shell:AppsFolder\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
    ]


def test_execute_reports_original_and_resolved_targets_on_open_failure(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")

    class FakeManager:
        def resolve(self, target):
            return ApplicationRecord(
                name=target,
                launch_target=r"shell:AppsFolder\Demo.App",
                provider="windows.start_apps",
            )

    def fail_open(target):
        raise OSError("native launch failed")

    monkeypatch.setattr(OpenApplicationTool, "_open", staticmethod(fail_open))

    result = OpenApplicationTool(FakeManager()).execute(make_request("demo"))

    assert result.success is False
    assert result.output == {
        "target": "demo",
        "resolved_target": r"shell:AppsFolder\Demo.App",
    }
    assert "native launch failed" in result.error


def test_unknown_windows_application_returns_failure(monkeypatch):
    monkeypatch.setattr(system.os, "name", "nt")
    monkeypatch.setattr(system.shutil, "which", lambda target: None)

    class MissingManager:
        def resolve(self, target):
            raise system.ApplicationResolutionError("No installed application matched")

    result = OpenApplicationTool(MissingManager()).execute(
        make_request("definitely-not-installed")
    )

    assert result.success is False
    assert "no installed application" in result.error.lower()


class FakeManager:
    def resolve(self, target):
        return ApplicationRecord(
            name=target,
            launch_target=target,
            provider="test",
        )
