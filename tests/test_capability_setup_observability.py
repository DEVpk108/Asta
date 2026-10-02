from core.autonomy.capability_setup import (
    CapabilitySetupManager,
    CapabilitySetupStatus,
)
from core.task_manager import TaskManager


class FakeMediaManager:
    def refresh_configuration(self):
        pass


class FailedResult:
    status = CapabilitySetupStatus.FAILED
    summary = "Browser launch failed."
    metadata = {"browser_automation": "error"}


def test_capability_setup_manager_surfaces_failure():
    messages = []
    events = []
    manager = CapabilitySetupManager(
        FakeMediaManager(),
        event_bus=type(
            "Bus",
            (),
            {
                "emit": lambda self, name, **payload: (
                    events.append((name, payload)),
                    messages.append(payload.get("text"))
                    if name == "assistant_sentence"
                    else None,
                )[-1]
            },
        )(),
        operators={"spotify": lambda: FailedResult()},
    )
    task = TaskManager().create("play hanuman chalisa on spotify", activate=False)

    assert manager.start(task, capability="spotify", step_id="step-2") is True

    import time
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if any(name == "capability_setup_completed" for name, _ in events):
            break
        time.sleep(0.01)

    assert any("Browser launch failed." in str(message) for message in messages)


def test_spotify_windows_chrome_finder_uses_known_install_locations(monkeypatch, tmp_path):
    from core.autonomy.spotify_setup import _PlaywrightSpotifyBrowser

    chrome_root = tmp_path / "ChromeRoot"
    executable = chrome_root / "Google" / "Chrome" / "Application" / "chrome.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("stub", encoding="utf-8")

    monkeypatch.setenv("PROGRAMFILES", str(chrome_root))
    monkeypatch.setenv("PROGRAMFILES(X86)", "")
    monkeypatch.setenv("LOCALAPPDATA", "")

    assert _PlaywrightSpotifyBrowser._find_windows_chrome() == str(executable)
