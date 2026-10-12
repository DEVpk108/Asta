import pytest


@pytest.fixture(autouse=True)
def _no_catalog_lookup(monkeypatch, tmp_path):
    """Tests never hit the network or write the user's music history."""
    monkeypatch.setenv("ASTA_CATALOG_RESOLVE", "0")
    monkeypatch.setenv("ASTA_MUSIC_HISTORY", str(tmp_path / "music_history.json"))
    # Never start Docker Desktop or build images on the developer's machine
    # from a test; self-repair tests opt back in with fakes.
    monkeypatch.setenv("ASTA_SANDBOX_SELF_REPAIR", "0")
    from core.media import catalog

    monkeypatch.setattr(catalog, "_resolver", None)
