import json

import pytest

import core.applications.manager as manager_module
from core.applications import ApplicationManager, ApplicationRecord, ApplicationResolutionError


def _empty_start_menu(monkeypatch):
    monkeypatch.setattr(
        manager_module.ApplicationManager,
        "_discover_start_menu",
        lambda self: [],
    )


def test_normalize_application_name():
    assert manager_module.normalize_application_name(" Visual-Studio_Code ") == "visual studio code"


def test_start_apps_discovers_whatsapp_and_file_explorer(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    _empty_start_menu(monkeypatch)

    payload = json.dumps(
        [
            {"Name": "WhatsApp", "AppID": "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"},
            {"Name": "File Explorer", "AppID": "Microsoft.Windows.Explorer"},
        ]
    )

    manager = ApplicationManager(powershell_runner=lambda _: payload)
    records = manager.refresh()

    whatsapp = manager.resolve("whatsapp")
    explorer = manager.resolve("file explorer")

    assert len(records) == 2
    assert whatsapp.provider == "windows.start_apps"
    assert whatsapp.app_id == "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"
    assert whatsapp.launch_target.endswith(whatsapp.app_id)
    assert explorer.name == "File Explorer"


def test_start_apps_accepts_single_json_object(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    _empty_start_menu(monkeypatch)

    payload = json.dumps({"Name": "Notepad", "AppID": "Notepad.App"})
    manager = ApplicationManager(powershell_runner=lambda _: payload)

    assert manager.resolve("notepad").name == "Notepad"


def test_discovery_falls_back_to_start_menu(monkeypatch, tmp_path):
    shortcut = tmp_path / "Programs" / "My Editor.lnk"
    shortcut.parent.mkdir()
    shortcut.write_text("placeholder")

    manager = ApplicationManager(
        powershell_runner=lambda _: (_ for _ in ()).throw(
            RuntimeError("PowerShell unavailable")
        )
    )
    monkeypatch.setattr(
        manager_module.ApplicationManager,
        "_discover_start_menu",
        lambda self: [
            ApplicationRecord(
                name="My Editor",
                launch_target=str(shortcut),
                provider="windows.start_menu",
                source="Start Menu",
            )
        ],
    )
    monkeypatch.setattr(manager_module.os, "name", "nt")

    assert manager.resolve("my editor").launch_target == str(shortcut)
    assert manager.last_error == "RuntimeError: PowerShell unavailable"


def test_discovery_supports_fuzzy_queries(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    _empty_start_menu(monkeypatch)

    payload = json.dumps(
        [
            {"Name": "Visual Studio Code", "AppID": "VSCode"},
            {"Name": "Google Chrome", "AppID": "Chrome"},
        ]
    )
    manager = ApplicationManager(powershell_runner=lambda _: payload)

    assert manager.resolve("google chrome").name == "Google Chrome"
    assert manager.resolve("chrome").name == "Google Chrome"


def test_unknown_application_is_not_resolved(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    _empty_start_menu(monkeypatch)

    manager = ApplicationManager(powershell_runner=lambda _: "[]")

    with pytest.raises(ApplicationResolutionError, match="No installed application"):
        manager.resolve("does-not-exist")


def test_discovery_supports_abbreviated_tokens(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    _empty_start_menu(monkeypatch)

    payload = json.dumps([
        {"Name": "Visual Studio Code", "AppID": "VSCode"},
    ])
    manager = ApplicationManager(powershell_runner=lambda _: payload)

    assert manager.resolve("vs code").name == "Visual Studio Code"
    assert manager.resolve("vscode").name == "Visual Studio Code"


def test_running_process_discovery_matches_application_path(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    start_apps = json.dumps(
        {"Name": "Free Download Manager", "AppID": "FreeDownloadManager.App"}
    )
    running_process = json.dumps(
        {
            "ProcessId": 4242,
            "Name": "fdm",
            "Path": r"C:\Program Files\Free Download Manager\fdm.exe",
            "WindowTitle": "Free Download Manager",
        }
    )

    def runner(command):
        if "Get-StartApps" in command:
            return start_apps
        return running_process

    manager = ApplicationManager(powershell_runner=runner)
    process = manager.resolve_running_process("free download manager")

    assert process.pid == 4242
    assert process.name == "fdm"
    assert process.executable_path.endswith(r"Free Download Manager\fdm.exe")


def test_running_process_discovery_accepts_compact_application_name(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")
    start_apps = json.dumps(
        {"Name": "Visual Studio Code", "AppID": "VSCode"}
    )
    running_process = json.dumps(
        {
            "ProcessId": 3131,
            "Name": "Code",
            "Path": r"C:\Program Files\Microsoft VS Code\Code.exe",
            "WindowTitle": "project - Visual Studio Code",
        }
    )

    def runner(command):
        if "Get-StartApps" in command:
            return start_apps
        return running_process

    manager = ApplicationManager(powershell_runner=runner)

    process = manager.resolve_running_process("vscode")

    assert process.pid == 3131
    assert process.name == "Code"
