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
