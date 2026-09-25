from pathlib import Path

from core.config import load_local_environment, update_local_environment


def test_load_local_environment_sets_missing_values(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "ASTA_TEST_ALPHA=one\n"
        "# comment\n"
        'ASTA_TEST_QUOTED="two"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ASTA_TEST_ALPHA", raising=False)
    monkeypatch.delenv("ASTA_TEST_QUOTED", raising=False)

    loaded = load_local_environment(path)

    assert loaded == {
        "ASTA_TEST_ALPHA": "one",
        "ASTA_TEST_QUOTED": "two",
    }


def test_update_local_environment_preserves_existing_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# keep me\n"
        "ASTA_TEST_ALPHA=old\n"
        "ASTA_TEST_OTHER=keep\n",
        encoding="utf-8",
    )

    update_local_environment(
        {"ASTA_TEST_ALPHA": "new", "ASTA_TEST_CLIENT": "abc123"},
        path,
    )

    assert path.read_text(encoding="utf-8") == (
        "# keep me\n"
        "ASTA_TEST_ALPHA=new\n"
        "ASTA_TEST_OTHER=keep\n"
        "\n"
        "ASTA_TEST_CLIENT=abc123\n"
    )
