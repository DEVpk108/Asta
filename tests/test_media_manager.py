from core.contracts import ToolRequest
from core.media import (
    MediaManager,
    MediaRequest,
    MediaResult,
    SpotifyProvider,
    parse_media_request,
)
from core.tools import MediaControlTool


def test_parse_media_play_query_and_provider():
    request = parse_media_request("Play Hanuman Chalisa on Spotify")
    assert request == MediaRequest(
        operation="play",
        query="hanuman chalisa",
        provider="spotify",
    )


def test_parse_media_controls_without_query():
    assert parse_media_request("Pause the music") == MediaRequest(
        operation="pause"
    )
    assert parse_media_request("Next track") == MediaRequest(
        operation="next"
    )
    assert parse_media_request("Go back") == MediaRequest(
        operation="previous"
    )


def test_media_manager_prefers_provider_that_supports_query():
    class FakeProvider:
        name = "fake"
        aliases = ()
        priority = 50

        def supports(self, request):
            return request.operation == "play" and bool(request.query)

        def execute(self, request):
            return MediaResult(
                success=True,
                provider=self.name,
                operation=request.operation,
                query=request.query,
                message="Fake provider handled it.",
            )

    manager = MediaManager(providers=(FakeProvider(),))
    result = manager.execute(
        MediaRequest(
            operation="play",
            query="test song",
            provider="fake",
        )
    )

    assert result.success is True
    assert result.provider == "fake"
    assert result.message == "Fake provider handled it."


def test_media_tool_builds_generic_media_capability():
    manager = MediaManager(providers=())
    tool = MediaControlTool(manager)

    request = ToolRequest(
        tool="media.control",
        arguments={
            "operation": "play",
            "query": "test song",
            "provider": "fake",
        },
        request_id="media-test",
    )

    result = tool.execute(request)
    assert result.success is False
    assert "Unknown media provider" in (result.error or "")



def test_media_manager_infers_provider_from_application_context():
    manager = MediaManager(providers=(
        SpotifyProvider(),
    ))

    assert manager.provider_for_application("Spotify") == "spotify"


def test_spotify_query_play_requires_authenticated_configuration(monkeypatch):
    monkeypatch.delenv("ASTA_SPOTIFY_CLIENT_ID", raising=False)
    provider = SpotifyProvider()

    result = provider.execute(
        MediaRequest(operation="play", query="hanuman chalisa")
    )

    assert result.success is False
    assert "ASTA_SPOTIFY_CLIENT_ID" in (result.error or "")



def test_parse_media_recovers_dropped_play_verb_before_known_provider():
    from core.media import parse_media_request

    request = parse_media_request(
        "Only Hanuman Chalisa on Spotify",
        known_providers=("spotify", "system"),
    )

    assert request == MediaRequest(
        operation="play",
        query="hanuman chalisa",
        provider="spotify",
    )


def test_media_manager_does_not_silently_default_query_playback_to_spotify(monkeypatch):
    monkeypatch.delenv("ASTA_MEDIA_DEFAULT_PROVIDER", raising=False)
    manager = MediaManager(providers=(SpotifyProvider(),))

    result = manager.execute(
        MediaRequest(operation="play", query="hanuman chalisa")
    )

    assert result.success is False
    assert "provider" in (result.error or "").lower()
