import json

import pytest

from core.media import catalog
from core.media.catalog import SongResolver, phonetic_key, similarity

SONGS = [
    {"title": "Baithi Hai", "artist": "Amit Trivedi, Amitabh Bhattacharya", "album": "Songs of Trance"},
    {"title": 'Tum Hi Ho (From "Aashiqui 2")', "artist": "Arijit Singh", "album": "Aashiqui 2"},
    {"title": "Tumhi Ho Bandhu", "artist": "Neeraj Shridhar", "album": "Cocktail"},
    {"title": "Raataan Lambiyan", "artist": "Jubin Nautiyal & Asees Kaur", "album": "Shershaah"},
]


def fake_fetch(term, **_kw):
    words = set(phonetic_key(term).split())
    return [s for s in SONGS if words & set(phonetic_key(s["title"]).split())][:20]


@pytest.fixture
def resolver(tmp_path):
    return SongResolver(fetch=fake_fetch, history_path=tmp_path / "h.json")


@pytest.mark.parametrize("heard", ["baiti hai", "baithee hai", "bethi hai", "Baithi Hai on apple music", "baithi hai on eppal music"])
def test_misspelled_hindi_title_snaps_to_catalog(resolver, heard):
    found = resolver.resolve(heard)
    assert found and found.query == "Baithi Hai" and found.source == "catalog"


def test_phonetic_similarity_ignores_spelling_but_not_words():
    assert phonetic_key("baithee") == phonetic_key("baiti")
    assert similarity("ratan lambiyan", "Raataan Lambiyan") > 0.9
    assert similarity("udit narayan songs", "Baithi Hai") < 0.4


def test_artist_in_query_is_kept_and_matched(resolver):
    assert resolver.resolve("baiti hai by amit").query == "Baithi Hai Amit Trivedi"


def test_featured_edition_suffix_is_removed_and_spacing_is_not_a_tie(resolver):
    assert resolver.resolve("tumhi ho").query == "Tum Hi Ho"


def test_generic_or_unknown_requests_are_left_alone(resolver):
    for query in ("some songs", "udit narayan songs", "zzzz qqqq"):
        assert resolver.resolve(query) is None


def test_offline_lookup_leaves_query_unchanged(tmp_path):
    def broken(*_a, **_k):
        raise OSError("no network")

    assert SongResolver(fetch=broken, history_path=tmp_path / "h.json").resolve("baiti hai") is None


def test_successful_plays_teach_the_misheard_spelling(tmp_path):
    path = tmp_path / "h.json"
    first = SongResolver(fetch=fake_fetch, history_path=path)
    found = first.resolve("baithi hain")
    first.note_success(found.query)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["songs"] and data["aliases"]

    def offline(*_a, **_k):
        raise OSError

    again = SongResolver(fetch=offline, history_path=path).resolve("baithi hain")
    assert again and again.title == "Baithi Hai" and again.source == "alias"


def test_unrelated_track_is_never_learned(tmp_path):
    resolver = SongResolver(fetch=fake_fetch, history_path=tmp_path / "h.json")
    resolver.note_success("baithi hai", "Completely Different Song", "Someone")
    assert not (tmp_path / "h.json").exists()


def test_planner_hook_rewrites_play_query(monkeypatch, resolver):
    monkeypatch.setenv("ASTA_CATALOG_RESOLVE", "1")
    monkeypatch.setattr(catalog, "_resolver", resolver)
    assert catalog.resolve_play_query("baiti hai") == "Baithi Hai"
    monkeypatch.setenv("ASTA_CATALOG_RESOLVE", "0")
    assert catalog.resolve_play_query("baiti hai") == "baiti hai"


def test_truncated_artist_name_still_matches(resolver):
    # Speech cut "Amit" to "Ami".
    found = resolver.resolve("baithi hai by ami")
    assert found.query == "Baithi Hai Amit Trivedi"


def test_ui_play_prefers_capitalised_song_rows_over_lowercase_suggestions():
    from core.tools import ui_play

    assert "-cmatch '[A-Z]'" in ui_play._FIND_AND_INVOKE
