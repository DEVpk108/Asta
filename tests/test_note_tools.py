from core.contracts import ToolRequest
from core.notes_manager import NotesManager
from core.tools import (
    CreateNoteTool,
    ListNotesTool,
    ReadNoteTool,
    SearchNotesTool,
)


def request(tool, **arguments):
    return ToolRequest(
        tool=tool,
        arguments=arguments,
        request_id="test-request",
    )


def test_create_note_tool(tmp_path):
    manager = NotesManager(tmp_path)
    tool = CreateNoteTool(manager)

    result = tool.execute(
        request(
            "notes.create_note",
            title="Ideas",
            content="Build a local notification system.",
        )
    )

    assert result.success is True
    assert result.output["title"] == "Ideas"


def test_read_note_tool(tmp_path):
    manager = NotesManager(tmp_path)
    manager.create("Remember to test voice interruption.", title="Voice Test")
    tool = ReadNoteTool(manager)

    result = tool.execute(
        request("notes.read_note", target="Voice Test")
    )

    assert result.success is True
    assert result.output["content"] == "Remember to test voice interruption."


def test_list_and_search_note_tools(tmp_path):
    manager = NotesManager(tmp_path)
    manager.create("Use local models first.", title="Architecture")
    manager.create("Test microphone interruption.", title="Voice")

    listed = ListNotesTool(manager).execute(
        request("notes.list_notes")
    )
    assert listed.success is True
    assert listed.output["count"] == 2

    searched = SearchNotesTool(manager).execute(
        request("notes.search_notes", query="microphone")
    )
    assert searched.success is True
    assert searched.output["count"] == 1
    assert searched.output["matches"][0]["title"] == "Voice"
