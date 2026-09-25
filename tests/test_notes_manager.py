from core.notes_manager import NotesManager


def test_notes_manager_creates_reads_lists_and_searches(tmp_path):
    manager = NotesManager(tmp_path)

    created = manager.create(
        "Test the A.S.T.A. barge-in path.",
        title="Barge In",
    )

    assert created["title"] == "Barge In"
    assert created["path"].endswith("barge-in.md")

    read = manager.read("barge in")
    assert read["title"] == "Barge In"
    assert read["content"] == "Test the A.S.T.A. barge-in path."

    notes = manager.list()
    assert notes == [{"title": "Barge In", "path": created["path"]}]

    matches = manager.search("barge-in")
    assert len(matches) == 1
    assert matches[0]["title"] == "Barge In"


def test_notes_manager_derives_title_and_appends(tmp_path):
    manager = NotesManager(tmp_path)

    created = manager.create("First line becomes the title.\nSecond line.")
    assert created["title"] == "First line becomes the title"

    updated = manager.append(created["title"], "Third line.")
    assert updated["updated"] is True

    read = manager.read(created["title"])
    assert "Second line." in read["content"]
    assert "Third line." in read["content"]


def test_notes_manager_rejects_duplicate_titles(tmp_path):
    manager = NotesManager(tmp_path)
    manager.create("One", title="Same")

    try:
        manager.create("Two", title="Same")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Duplicate notes should be rejected")
