from core.contracts import IntentType
from core.intent_router import IntentRouter
from core.media import MediaRequest, parse_media_request


def test_parse_media_recovers_extra_stt_on_before_known_provider():
    request = parse_media_request(
        "Play on Manchalisa on Spotify",
        known_providers=("spotify", "system"),
    )

    assert request == MediaRequest(
        operation="play",
        query="manchalisa",
        provider="spotify",
    )


def test_intent_router_uses_known_provider_for_noisy_media_suffix():
    router = IntentRouter(media_providers=("spotify", "system"))

    result = router.analyze("Play on Manchalisa on Spotify")

    assert result.intent == IntentType.COMMAND
    assert result.entities == {
        "action": "media",
        "operation": "play",
        "query": "manchalisa",
        "provider": "spotify",
    }
