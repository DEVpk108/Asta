"""Recognise "songs by <artist>" requests and repair garbled artist names.

"play some Udit Narayan songs" means the artist, not a track literally
called "Udit Narayan songs". Speech recognition also glues or garbles
names ("uditnarayan", "udnarayan"), so known names are snapped back.
Add your own names in data/artists.txt (one per line).
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path

KNOWN_ARTISTS = tuple(
    name.strip()
    for name in """
    udit narayan|kumar sanu|alka yagnik|sonu nigam|arijit singh|shreya ghoshal|lata mangeshkar
    kishore kumar|mohammed rafi|mukesh|asha bhosle|jagjit singh|nusrat fateh ali khan|atif aslam
    rahat fateh ali khan|sunidhi chauhan|shaan|kk|mohit chauhan|lucky ali|kailash kher|shankar mahadevan
    a r rahman|ar rahman|pritam|vishal shekhar|amit trivedi|jubin nautiyal|armaan malik|neha kakkar
    darshan raval|b praak|badshah|yo yo honey singh|honey singh|diljit dosanjh|ap dhillon|karan aujla
    sidhu moose wala|guru randhawa|mika singh|anuv jain|prateek kuhad|king|javed ali|sukhwinder singh
    hariharan|kavita krishnamurthy|anuradha paudwal|sadhana sargam|abhijeet|palak muchhal|tulsi kumar
    vishal mishra|sachet tandon|stebin ben|papon|shilpa rao|jasleen royal|ankit tiwari|mithoon
    taylor swift|ed sheeran|the weeknd|drake|eminem|coldplay|imagine dragons|justin bieber
    billie eilish|ariana grande|bts|dua lipa|bruno mars|post malone|adele|rihanna|beyonce
    """.replace("\n", "|").split("|")
    if name.strip()
)
_EXTRA_FILE = Path(__file__).resolve().parents[2] / "data" / "artists.txt"

_MUSIC_WORDS = r"(?:songs?|music|tracks?|hits?|gaane|gane|gaana|gana|numbers?|playlist)"
_LEAD = r"(?:(?:some|a\s+few|few|any|the|my|best|top|popular|old|new|latest|famous)\s+)*"
_BY_ARTIST = re.compile(
    rf"^(?:{_LEAD}(?:[a-z0-9']+\s+){{0,2}}?{_MUSIC_WORDS}\s+(?:by|of|from|sung\s+by)|"
    rf"(?:the\s+)?(?:best|greatest\s+hits|hits)\s+of)\s+(?P<artist>.+?)(?:\s+{_MUSIC_WORDS})?$",
    re.IGNORECASE,
)
_ARTIST_SONGS = re.compile(
    rf"^{_LEAD}(?P<artist>.+?)(?P<possessive>'s|\s+ke|\s+ki|\s+wale|\s+waale)?\s+{_MUSIC_WORDS}$",
    re.IGNORECASE,
)


def known_artists() -> tuple[str, ...]:
    names = list(KNOWN_ARTISTS)
    path = Path(os.getenv("ASTA_ARTISTS_FILE", str(_EXTRA_FILE)))
    try:
        names += [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        pass
    return tuple(names)


def _squash(text: str) -> str:
    value = re.sub(r"[^a-z]", "", str(text or "").lower())
    for long, short in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u")):
        value = value.replace(long, short)
    return value


def _match_artist(span: str, *, minimum: float = 0.82) -> tuple[str | None, float]:
    squashed = _squash(span)
    if len(squashed) < 2:
        return None, 0.0
    best, best_score = None, 0.0
    for name in known_artists():
        target = _squash(name)
        if squashed == target:
            return name, 1.0
        if len(target) < 8 or len(squashed) < 6 or abs(len(target) - len(squashed)) > (3 if minimum >= 0.8 else 4):
            continue  # short names ("kk", "king") must match exactly
        score = SequenceMatcher(None, squashed, target, autojunk=False).ratio()
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= minimum else (None, 0.0)


_GLUE_WORDS = frozenset(
    "a an the some sam sum any by of from ke ki ka wale waale and songs song music gaane gane hits on in".split()
)


def snap_artist_names(query: str) -> str:
    """"sam uditnarayan music" -> "sam udit narayan music"."""
    tokens = str(query or "").split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        best = (None, 0.0, 1)
        for span in (1, 2, 3):
            chunk = tokens[i:i + span]
            if len(chunk) < span or chunk[0].lower() in _GLUE_WORDS or chunk[-1].lower() in _GLUE_WORDS:
                continue
            name, score = _match_artist(" ".join(chunk))
            if name and score > best[1]:
                best = (name, score, span)
        if best[0]:
            out.append(best[0])
            i += best[2]
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def artist_request(query: str) -> str | None:
    """Artist name when the query asks for an artist's songs, else None."""
    value = " ".join(str(query or "").lower().split())
    # "sam"/"sum" is a misheard "some".
    value = re.sub(r"^(?:sam|sum|som)\s+", "some ", value)
    match = _BY_ARTIST.match(value)
    if match:
        artist = match.group("artist").strip()
        name, _ = _match_artist(artist)
        return name or artist
    match = _ARTIST_SONGS.match(value)
    if match:
        artist = match.group("artist").strip()
        name, _ = _match_artist(artist)
        if name:
            return name
        if match.group("possessive") and len(artist.split()) <= 4:
            # "atif aslam ke gaane" / "taylor's songs" name a person even if unknown.
            return artist
    return None


def best_artist(candidates, *, minimum: float = 0.7) -> str | None:
    """Known artist closest to any candidate spelling (explicit "by X" slot)."""
    best, best_score = None, 0.0
    for candidate in candidates:
        name, score = _match_artist(candidate, minimum=minimum)
        if name and score > best_score:
            best, best_score = name, score
    return best
