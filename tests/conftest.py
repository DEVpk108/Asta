import pytest


@pytest.fixture(autouse=True)
def _no_catalog_lookup(monkeypatch, tmp_path):
    """Tests never hit the network or write the user's music history."""
    monkeypatch.setenv("ASTA_CATALOG_RESOLVE", "0")
    monkeypatch.setenv("ASTA_MUSIC_HISTORY", str(tmp_path / "music_history.json"))
    from core.media import catalog

    monkeypatch.setattr(catalog, "_resolver", None)
