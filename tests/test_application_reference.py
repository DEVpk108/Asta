import json

import pytest

import core.applications.manager as manager_module
from core.applications import ApplicationManager, ApplicationRecord, ApplicationResolutionError


def test_recent_application_reference_resolves_to_last_opened_application():
    manager = ApplicationManager()
    record = ApplicationRecord(
        name="Spotify",
        launch_target=r"shell:AppsFolder\Spotify.App",
        provider="windows.start_apps",
        app_id="Spotify.App",
    )

    manager.remember_opened(record)

    assert manager.last_opened_application is record
    assert manager.resolve_reference("it") == "Spotify"
    assert manager.resolve_reference("that") == "Spotify"


def test_application_reference_without_recent_app_is_rejected():
    manager = ApplicationManager()

    with pytest.raises(ApplicationResolutionError, match="No recent application reference"):
        manager.resolve_reference("it")


def test_running_process_reference_uses_last_opened_application(monkeypatch):
    monkeypatch.setattr(manager_module.os, "name", "nt")

    start_apps = json.dumps(
        {"Name": "Spotify", "AppID": "Spotify.App"}
    )
    running_process = json.dumps(
        {
            "ProcessId": 4242,
            "Name": "Spotify",
            "Path": r"C:\Program Files\Spotify\Spotify.exe",
            "WindowTitle": "Spotify",
        }
    )

    def runner(command):
        if "Get-StartApps" in command:
            return start_apps
        return running_process

    manager = ApplicationManager(powershell_runner=runner)
    manager.remember_opened(manager.resolve("spotify"))

    process = manager.resolve_running_process("it")

    assert process.pid == 4242
    assert process.name == "Spotify"
