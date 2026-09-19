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
                launch_target="shell:AppsFolder\Demo.App",
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
    assert "not found" in result.error.lower()


class FakeManager:
    def resolve(self, target):
        return ApplicationRecord(
            name=target,
            launch_target=target,
            provider="test",
        )
