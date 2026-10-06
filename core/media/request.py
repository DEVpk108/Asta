from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_MEDIA_OPERATION_ALIASES = {
    "play": "play",
    "resume": "play",
    "continue": "play",
    "start": "play",
    "listen": "play",
    "listen to": "play",
    "put on": "play",
    "pause": "pause",
    "pause playback": "pause",
    "toggle": "toggle",
    "toggle playback": "toggle",
    "next": "next",
    "next track": "next",
    "next song": "next",
    "skip": "next",
    "skip track": "next",
    "skip song": "next",
    "previous": "previous",
    "previous track": "previous",
    "previous song": "previous",
    "last track": "previous",
    "go back": "previous",
    "stop": "stop",
    "stop music": "stop",
    "stop playback": "stop",
}


@dataclass(frozen=True, slots=True)
class MediaRequest:
    operation: str
    query: str | None = None
    provider: str | None = None

    def to_entities(self) -> dict[str, Any]:
        entities: dict[str, Any] = {
            "action": "media",
            "operation": self.operation,
        }
        if self.query:
            entities["query"] = self.query
        if self.provider:
            entities["provider"] = self.provider
        return entities


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split()).rstrip(" .!?;:")


def _strip_polite_leads(value: str) -> str:
    text = value
    changed = True
    leads = (
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
        "okay ",
        "ok ",
        "hey ",
    )
    while changed:
        changed = False
        for lead in leads:
            if text.startswith(lead):
                text = text[len(lead):].strip()
                changed = True
                break
    return text


def _extract_known_provider(normalized: str, known_providers) -> tuple[str, str]:
    """Strip a recognized trailing provider without greedily consuming the query."""
    if not known_providers:
        return "", normalized

    candidates = sorted(
        {
            _normalize(str(name))
            for name in known_providers
            if str(name).strip()
        },
        key=len,
        reverse=True,
    )

    for provider in candidates:
        if not provider:
            continue
        marker = f" on {provider}"
        if normalized.endswith(marker):
            remaining = normalized[: -len(marker)].strip()
            if remaining:
                return provider, remaining

    return "", normalized


def _clean_play_query(query: str) -> str:
    """Normalize common STT filler around a media play query."""
    cleaned = query.strip(" ,.-")
    for prefix in (
        "play ",
        "resume ",
        "continue ",
        "listen to ",
        "put on ",
        "only ",
        "just ",
        "please ",
        "okay ",
        "ok ",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ,.-")
            break

    # Some STT outputs insert an extra preposition, e.g.
    # "play on hanuman chalisa on spotify".
    if cleaned.startswith("on "):
        cleaned = cleaned[3:].strip(" ,.-")

    return cleaned


# Music apps ASTA has no integration for; requests fall back to Spotify.
UNSUPPORTED_MUSIC_APPS = frozenset(
    {"amazon music", "jiosaavn", "saavn", "gaana", "wynk", "wynk music",
     "soundcloud", "deezer", "tidal", "pandora"}
)


_APP_NAMES = r"apple\s+music|itunes|spotify|youtube\s+music|youtube|amazon\s+music|jiosaavn|saavn|gaana|wynk(?:\s+music)?|soundcloud|deezer|tidal|pandora"
_APP_BEFORE_ARTIST = re.compile(
    rf"^(?P<head>.+?)\s+(?P<prep>on|in)\s+(?P<app>{_APP_NAMES})\s+(?P<by>by\s+\S.*?)[\s.!?]*$",
    re.IGNORECASE,
)


def app_last(text: str) -> str:
    """"play X on Apple Music by Amit" -> "play X by Amit on Apple Music".

    The app is always parsed from the end of the sentence; an artist said
    after it would otherwise glue onto the app name or the title.
    """
    match = _APP_BEFORE_ARTIST.match(str(text or "").strip())
    if not match:
        return text
    return f"{match.group('head')} {match.group('by')} {match.group('prep')} {match.group('app')}"


def parse_media_request(
    text: str,
    *,
    known_providers=None,
) -> MediaRequest | None:
    normalized = _strip_polite_leads(_normalize(app_last(_normalize(text))))
    if not normalized:
        return None

    provider = None
    if known_providers:
        provider, remaining = _extract_known_provider(
            normalized,
            known_providers,
        )
        if provider:
            normalized = remaining
        else:
            # A music app ASTA has no integration for ("on apple music") is
            # still the app, not part of the song title; the planner falls
            # back to the default app and says so.
            other = re.search(
                rf"\s+(?:on|in)\s+(?P<provider>youtube\s+music|youtube|{'|'.join(re.escape(n) for n in sorted(UNSUPPORTED_MUSIC_APPS, key=len, reverse=True))})$",
                normalized,
                re.IGNORECASE,
            )
            if other:
                provider = other.group("provider").lower()
                normalized = normalized[:other.start()].strip()
    else:
        provider_match = re.search(
            r"\s+on\s+(?P<provider>[a-z0-9][a-z0-9 ._-]*)$",
            normalized,
            re.IGNORECASE,
        )
        if provider_match:
            provider = provider_match.group("provider").strip()
            normalized = normalized[:provider_match.start()].strip()
            if not provider:
                provider = None

    compact = normalized

    if provider and known_providers:
        normalized_providers = {
            _normalize(str(name))
            for name in known_providers
            if str(name).strip()
        }

        # Whisper can replace or drop the short command verb. When the
        # utterance has a known media provider suffix, treat the remaining
        # phrase as a play query. This is intentionally provider-agnostic:
        # the provider registry decides what names are recognized.
        if provider in normalized_providers and compact:
            cleaned_query = _clean_play_query(compact)
            if cleaned_query:
                return MediaRequest(
                    operation="play",
                    query=cleaned_query,
                    provider=provider,
                )

    if compact in _MEDIA_OPERATION_ALIASES:
        return MediaRequest(
            operation=_MEDIA_OPERATION_ALIASES[compact],
            provider=provider,
        )

    for phrase in sorted(_MEDIA_OPERATION_ALIASES, key=len, reverse=True):
        if phrase in {"play", "resume", "continue", "start", "listen", "listen to", "put on"}:
            continue
        prefix = phrase + " "
        if compact.startswith(prefix):
            remainder = compact[len(prefix):].strip()
            if remainder in {"music", "the music", "playback", "the playback", "media"}:
                remainder = None
            if remainder:
                return MediaRequest(
                    operation=_MEDIA_OPERATION_ALIASES[phrase],
                    provider=provider,
                    query=remainder,
                )
            return MediaRequest(
                operation=_MEDIA_OPERATION_ALIASES[phrase],
                provider=provider,
            )

    for phrase in ("listen to", "put on", "resume", "continue", "start", "play"):
        prefix = phrase + " "
        if compact.startswith(prefix):
            query = compact[len(prefix):].strip(" ,.-")
            return MediaRequest(
                operation="play",
                query=query or None,
                provider=provider,
            )

    return None
