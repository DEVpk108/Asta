"""Snap a heard song title to a real catalog title.

Speech recognition mangles Hindi song names ("baiti hai", "bethi hai"). Instead
of trusting the transcript, look the query up in Apple's public iTunes Search
API (no key needed; the catalog Apple Music and most apps share), score every
candidate by how it *sounds* against the heard text, and use the real title.
Songs the user has actually played before are remembered, including the
misheard spellings that led to them, so repeat misrecognitions fix themselves.

The resolver is deliberately conservative: when nothing is a clear phonetic
match (artist-only requests, generic phrases, offline), the query is returned
unchanged.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

ITUNES_URL = "https://itunes.apple.com/search"
_HISTORY_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "music_history.json"
_APP_TAIL = re.compile(
    r"\s+(?:on|in|using|with)\s+(?:the\s+)?(?:\w+\s+){0,2}(?:music|musik|muzic|spotify|youtube|tube)\s*$",
    re.IGNORECASE,
)
_BY = re.compile(r"\s+(?:by|bay|bye|buy)\s+", re.IGNORECASE)
_STRIP_TITLE = re.compile(r"\s*[\(\[].*?[\)\]]|\s+-\s+(?:from|feat|with|remix|reprise|lofi|lo-fi)\b.*$", re.IGNORECASE)
_ARTIST_SPLIT = re.compile(r"\s*(?:,|&|\band\b|\bfeat\.?|\bft\.?|\bwith\b)\s*", re.IGNORECASE)
_GENERIC_WORDS = {"song", "songs", "music", "track", "tracks", "playlist", "album", "some", "any", "something"}


# -- phonetics ---------------------------------------------------------------

def _ascii(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def phonetic_key(text: str) -> str:
    """Spelling-insensitive form of romanised Hindi/English: "baithee" == "baiti"."""
    t = re.sub(r"[^a-z0-9\s]", " ", _ascii(text))
    for pattern, repl in (
        (r"ch", "C"), (r"sh", "S"), (r"ph", "f"), (r"([kgtdbj])h", r"\1"),
        (r"w", "v"), (r"z", "j"), (r"q", "k"), (r"x", "ks"),
        (r"c(?=[eiy])", "s"), (r"c", "k"),
        (r"aa+", "a"), (r"(?:ee+|ii+)", "i"), (r"(?:oo+|uu+)", "u"),
        (r"(?:ai|ay|ei|ey|ae)", "e"), (r"(?:au|ou|aw|ow)", "o"),
        (r"y", "i"), (r"(.)\1+", r"\1"),
    ):
        t = re.sub(pattern, repl, t)
    return " ".join(t.split())


def _skeleton(key: str) -> str:
    return re.sub(r"[aeiou\s]", "", key)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def similarity(heard: str, title: str) -> float:
    """0..1 phonetic similarity: spelling-normalised text blended with its consonant skeleton."""
    ka, kb = phonetic_key(heard), phonetic_key(title)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    whole = _ratio(ka.replace(" ", ""), kb.replace(" ", ""))
    bones = _ratio(_skeleton(ka), _skeleton(kb))
    return 0.55 * whole + 0.45 * bones


def clean_title(title: str) -> str:
    return " ".join(_STRIP_TITLE.sub("", str(title or "")).split())


def primary_artist(artist: str) -> str:
    parts = [p for p in _ARTIST_SPLIT.split(str(artist or "")) if p.strip()]
    return parts[0].strip() if parts else ""


# -- catalog + history ---------------------------------------------------------

def fetch_itunes(term: str, *, country: str = "IN", limit: int = 20, timeout: float = 2.5) -> list[dict[str, Any]]:
    import requests

    response = requests.get(
        ITUNES_URL,
        params={"term": term, "entity": "song", "media": "music", "country": country, "limit": limit},
        timeout=timeout,
    )
    response.raise_for_status()
    return [
        {"title": r.get("trackName", ""), "artist": r.get("artistName", ""), "album": r.get("collectionName", "")}
        for r in (response.json().get("results") or [])
        if r.get("trackName")
    ]


@dataclass
class Resolution:
    heard: str
    title: str
    artist: str
    score: float
    source: str  # "catalog" | "history" | "alias"
    query: str = ""  # what the rest of ASTA should search for
    alternatives: list[str] = field(default_factory=list)


class SongResolver:
    SNAP_THRESHOLD = 0.82
    TIE_MARGIN = 0.03

    def __init__(
        self,
        fetch: Callable[..., list[dict[str, Any]]] | None = None,
        history_path: Path | str | None = None,
        timeout: float | None = None,
    ):
        self.fetch = fetch or fetch_itunes
        self.history_path = Path(history_path or os.getenv("ASTA_MUSIC_HISTORY") or _HISTORY_DEFAULT)
        self.timeout = float(timeout if timeout is not None else os.getenv("ASTA_CATALOG_TIMEOUT", "2.5"))
        self.country = os.getenv("ASTA_CATALOG_COUNTRY", "IN")
        self._lock = threading.Lock()
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._history: dict[str, Any] | None = None
        self.last: Resolution | None = None

    # history ------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if self._history is None:
            try:
                data = json.loads(self.history_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            self._history = data if isinstance(data, dict) and isinstance(data.get("songs"), dict) else {"songs": {}, "aliases": {}}
            self._history.setdefault("aliases", {})
        return self._history

    def _save(self) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(self._history, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as exc:  # history is a nicety, never fatal
            print(f"[Catalog] Could not save music history: {exc}", flush=True)

    def note_success(self, query: str, title: str = "", artist: str = "") -> None:
        """A song really played: remember it and the spelling that led to it."""
        with self._lock:
            last = self.last
            if not title and last and query and query.strip().lower() == last.query.strip().lower():
                title, artist, heard = last.title, last.artist, last.heard
            elif title:
                heard = last.heard if last and last.query.strip().lower() == str(query).strip().lower() else query
            else:
                return
            title = clean_title(title)
            if not title:
                return
            # Never learn a song the app played that has nothing to do with what was asked.
            if similarity(self._split(heard)[0], title) < 0.6:
                return
            history = self._load()
            key = phonetic_key(title)
            entry = history["songs"].setdefault(key, {"title": title, "artist": primary_artist(artist), "plays": 0})
            entry["plays"] += 1
            entry["last"] = int(time.time())
            if artist and not entry.get("artist"):
                entry["artist"] = primary_artist(artist)
            heard_key = phonetic_key(self._split(heard)[0])
            if heard_key and heard_key != key:
                history["aliases"][heard_key] = key
            self._save()

    # resolution ----------------------------------------------------------
    @staticmethod
    def _split(query: str) -> tuple[str, str]:
        """("baithi hai by amit on apple music") -> ("baithi hai", "amit")."""
        text = _APP_TAIL.sub("", " ".join(str(query or "").split()))
        title, artist = (_BY.split(text, maxsplit=1) + [""])[:2]
        return title.strip(), artist.strip()

    def _search(self, term: str) -> list[dict[str, Any]]:
        if term in self._cache:
            return self._cache[term]
        try:
            rows = self.fetch(term, country=self.country, timeout=self.timeout)
        except Exception as exc:
            print(f"[Catalog] Lookup failed for {term!r}: {type(exc).__name__}", flush=True)
            rows = []
        self._cache[term] = rows
        return rows

    def _candidates(self, title: str) -> list[dict[str, Any]]:
        tokens = title.split()
        terms = [title]
        simple = phonetic_key(title)
        if simple and simple != title.lower():
            terms.append(simple)
        if len(tokens) > 2:
            terms.append(" ".join(tokens[:2]))
        terms = list(dict.fromkeys(t for t in terms if t))
        with ThreadPoolExecutor(max_workers=len(terms)) as pool:
            batches = list(pool.map(self._search, terms))
        seen, merged = set(), []
        for batch in batches:
            for rank, row in enumerate(batch):
                ident = (phonetic_key(row["title"]).replace(" ", ""), phonetic_key(primary_artist(row["artist"])).replace(" ", ""))
                if ident not in seen:
                    seen.add(ident)
                    merged.append({**row, "rank": rank})
        return merged

    def resolve(self, query: str) -> Resolution | None:
        title_part, artist_part = self._split(query)
        words = [w for w in re.findall(r"[\w']+", title_part.lower()) if w not in _GENERIC_WORDS]
        if not title_part or not words:
            return None
        history = self._load()
        heard_key = phonetic_key(title_part)

        alias = history["aliases"].get(heard_key)
        if alias and alias in history["songs"]:
            entry = history["songs"][alias]
            return self._finish(query, entry["title"], entry.get("artist", ""), 1.0, "alias", artist_part, [])

        scored: list[tuple[float, dict[str, Any], str]] = []
        for key, entry in history["songs"].items():
            s = similarity(title_part, entry["title"]) + min(0.06, 0.02 * entry.get("plays", 1))
            scored.append((s, {"title": entry["title"], "artist": entry.get("artist", ""), "rank": 0}, "history"))
        for row in self._candidates(title_part):
            title = clean_title(row["title"])
            s = similarity(title_part, title) + 0.03 * (1 - row["rank"] / 20)
            if artist_part:
                # "by ami" / "by amit" should match "Amit Trivedi": try the whole
                # name and each of its words.
                name = primary_artist(row["artist"])
                a = max([similarity(artist_part, name)] + [similarity(artist_part, w) for w in name.split()])
                s += 0.05 if a >= 0.7 else -0.1
            scored.append((s, {**row, "title": title}, "catalog"))
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best, source = scored[0]
        if best_score < self.SNAP_THRESHOLD:
            return None
        best_key = phonetic_key(best["title"]).replace(" ", "")
        rivals = [
            row["title"]
            for s, row, _ in scored[1:]
            if phonetic_key(row["title"]).replace(" ", "") != best_key and best_score - s < self.TIE_MARGIN
        ]
        if rivals:
            print(f"[Catalog] {query!r} is ambiguous: {best['title']!r} vs {rivals[:2]}; keeping the heard text.", flush=True)
            return None
        return self._finish(query, best["title"], best.get("artist", ""), min(best_score, 1.0), source, artist_part, [])

    def _finish(self, heard, title, artist, score, source, artist_hint, alternatives) -> Resolution:
        query = title
        if artist_hint and artist:
            query = f"{title} {primary_artist(artist)}"
        result = Resolution(heard=self._split(heard)[0], title=title, artist=artist, score=score,
                            source=source, query=query, alternatives=alternatives)
        self.last = result
        return result


_resolver: SongResolver | None = None


def get_resolver() -> SongResolver:
    global _resolver
    if _resolver is None:
        _resolver = SongResolver()
    return _resolver


def enabled() -> bool:
    return os.getenv("ASTA_CATALOG_RESOLVE", "1").strip().lower() not in {"0", "false", "no", "off"}


def resolve_play_query(query: str) -> str:
    """The query ASTA should play: a real catalog title when one clearly matches, else unchanged."""
    if not enabled() or not str(query or "").strip():
        return query
    try:
        found = get_resolver().resolve(query)
    except Exception as exc:
        print(f"[Catalog] Resolver error: {type(exc).__name__}: {exc}", flush=True)
        return query
    if not found:
        return query
    print(
        f"[Catalog] {query!r} -> {found.query!r} (score={found.score:.2f}, {found.source}"
        f"{', ' + primary_artist(found.artist) if found.artist else ''})",
        flush=True,
    )
    return found.query


def note_play_success(query: str, title: str = "", artist: str = "") -> None:
    if not enabled():
        return
    try:
        get_resolver().note_success(query, title, artist)
    except Exception as exc:
        print(f"[Catalog] Could not record play: {type(exc).__name__}: {exc}", flush=True)
