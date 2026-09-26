import os

from core.autonomy.spotify_setup import (
    SpotifySetupOperator,
    SpotifySetupStatus,
)


def test_spotify_setup_persists_client_id(monkeypatch, tmp_path):
    calls = []

    operator = SpotifySetupOperator(
        config_path=tmp_path / ".env",
        browser_factory=lambda **kwargs: type(
            "Browser",
            (),
            {"run": lambda self: "abc123clientid123456789012345678"},
        )(),
        notify=calls.append,
    )

    monkeypatch.delenv("ASTA_SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.setenv(
        "ASTA_SPOTIFY_REDIRECT_URI",
        "http://127.0.0.1:8765/callback",
    )

    result = operator.run()

    assert result.status is SpotifySetupStatus.COMPLETED
    assert result.client_id == "abc123clientid123456789012345678"
    assert "ASTA_SPOTIFY_CLIENT_ID=abc123clientid123456789012345678" in (
        tmp_path / ".env"
    ).read_text(encoding="utf-8")
    assert calls == [
        (
            "Spotify needs initial developer setup. I’m opening the Spotify Developer "
            "Dashboard and will continue automatically after any required sign-in."
        ),
        "Spotify developer setup is complete. I’m resuming the original task.",
    ]


def test_spotify_setup_missing_browser_uses_user_boundary(monkeypatch, tmp_path):
    operator = SpotifySetupOperator(config_path=tmp_path / ".env")

    def missing_browser(*args, **kwargs):
        raise ImportError("playwright")

    monkeypatch.setattr(
        operator,
        "_run_browser_setup",
        missing_browser,
    )
    opened = []
    monkeypatch.setattr(
        "core.autonomy.spotify_setup.webbrowser.open",
        lambda url, **_kwargs: opened.append(url),
    )

    result = operator.run()

    assert result.status is SpotifySetupStatus.WAITING_FOR_USER
    assert result.requires_user is True
    assert opened == [
        "https://developer.spotify.com/dashboard"
    ]
