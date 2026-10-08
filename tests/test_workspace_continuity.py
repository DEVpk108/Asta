from __future__ import annotations

import json
from pathlib import Path

from core import Kernel
from core.context_builder import ContextBuilder
from core.contracts import ToolRequest
from core.tools import ReadWorkspaceFileTool, WriteWorkspaceFileTool
from core.workspace_runtime import WorkspaceRuntimeModule
from core.workspace_store import WorkspaceStateStore


def test_workspace_store_persists_only_safe_project_context(tmp_path):
    path = tmp_path / "state" / "workspace.json"
    store = WorkspaceStateStore(path)
    store.save(
        {
            "project_name": "Example",
            "project_path": str(tmp_path / "project"),
            "repository": "https://user:secret@github.com/acme/example.git?token=hidden",
            "branch": "feature/context",
            "active_files": ["src/main.py", ".env", "../outside.py"],
            "recent_files": ["README.md", "src/main.py", "C:\\Users\\PK\\private.py"],
            "environment": {"python_executable": "secret-local-path"},
            "hardware": {"processor": "private-device-detail"},
        }
    )

    loaded = store.load()
    raw = path.read_text(encoding="utf-8")

    assert loaded == {
        "project_name": "Example",
        "project_path": str(tmp_path / "project"),
        "repository": "https://github.com/acme/example.git",
        "branch": "feature/context",
        "active_files": ["src/main.py"],
        "recent_files": ["README.md", "src/main.py"],
    }
    assert "secret" not in raw
    assert "hidden" not in raw
    assert "python_executable" not in raw
    assert "processor" not in raw


def test_workspace_store_ignores_malformed_paths_without_blocking_startup(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_name": "stale",
                "project_path": "~asta-user-that-does-not-exist/project",
                "active_files": ["src/main.py"],
            }
        ),
        encoding="utf-8",
    )

    assert WorkspaceStateStore(path).load() == {
        "project_name": "stale",
        "active_files": ["src/main.py"],
        "recent_files": [],
    }


def test_runtime_repository_identity_drops_userinfo_query_and_fragment():
    value = "https://user:secret@github.com/acme/example.git?token=hidden#fragment"
    scp_value = "token-user@github.com:acme/example.git"

    assert WorkspaceRuntimeModule._normalize_repository(value) == (
        "https://github.com/acme/example.git"
    )
    assert WorkspaceRuntimeModule._normalize_repository(scp_value) == (
        "github.com:acme/example.git"
    )


def test_workspace_runtime_restores_existing_project_and_recent_files(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "remembered-project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("print('main')\n", encoding="utf-8")
    (project / "README.md").write_text("Project notes\n", encoding="utf-8")
    state_path = tmp_path / "workspace.json"
    WorkspaceStateStore(state_path).save(
        {
            "project_name": "remembered-project",
            "project_path": str(project),
            "repository": "https://github.com/acme/remembered-project.git",
            "branch": "main",
            "active_files": ["src/main.py"],
            "recent_files": ["src/main.py", "README.md"],
        }
    )
    monkeypatch.delenv("ASTA_WORKSPACE_PATH", raising=False)
    kernel = Kernel()
    module = WorkspaceRuntimeModule(kernel, state_path=state_path)
    module._git_value = lambda cwd, command: (
        "https://github.com/acme/remembered-project.git"
        if command[:3] == ["git", "config", "--get"]
        else "main"
    )

    module.initialize()
    try:
        state = kernel.workspace_manager.snapshot()
        assert state["project_path"] == str(project)
        assert state["repository"] == "https://github.com/acme/remembered-project.git"
        assert state["branch"] == "main"
        assert state["active_files"] == ["src/main.py"]
        assert state["recent_files"] == ["src/main.py", "README.md"]
        assert state["environment"]["python"]
        context = ContextBuilder(kernel).build("What project is this?")
        assert context.workspace["project_path"] == str(project)
        assert context.workspace["recent_files"] == ["src/main.py", "README.md"]

        read = ReadWorkspaceFileTool(kernel.workspace_manager).execute(
            ToolRequest(
                tool="filesystem.read_file",
                arguments={"path": "README.md"},
                request_id="continuity-read",
            )
        )
        assert read.success
        assert kernel.workspace_manager.snapshot()["recent_files"] == [
            "README.md",
            "src/main.py",
        ]

        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert "environment" not in saved
        assert "hardware" not in saved
        assert saved["recent_files"][0] == "README.md"
    finally:
        module.shutdown()


def test_explicit_workspace_path_overrides_saved_project_and_clears_file_history(
    tmp_path,
    monkeypatch,
):
    saved_project = tmp_path / "saved"
    explicit_project = tmp_path / "explicit"
    saved_project.mkdir()
    explicit_project.mkdir()
    state_path = tmp_path / "workspace.json"
    WorkspaceStateStore(state_path).save(
        {
            "project_name": "saved",
            "project_path": str(saved_project),
            "repository": "https://github.com/acme/saved.git",
            "branch": "main",
            "active_files": ["old.py"],
            "recent_files": ["old.py"],
        }
    )
    monkeypatch.setenv("ASTA_WORKSPACE_PATH", str(explicit_project))
    kernel = Kernel()
    module = WorkspaceRuntimeModule(kernel, state_path=state_path)
    module._git_value = lambda cwd, command: ""

    module.initialize()
    try:
        state = kernel.workspace_manager.snapshot()
        assert state["project_path"] == str(explicit_project)
        assert state["project_name"] == explicit_project.name
        assert state["active_files"] == []
        assert state["recent_files"] == []
    finally:
        module.shutdown()


def test_workspace_manager_clears_file_history_when_project_path_changes():
    kernel = Kernel()
    manager = kernel.workspace_manager
    manager.update_project(path="C:/Projects/first")
    manager.set_active_files(["src/first.py"])
    manager.set_recent_files(["src/first.py"])

    manager.update_project(path="C:/Projects/second")

    state = manager.snapshot()
    assert state["project_path"] == "C:/Projects/second"
    assert state["active_files"] == []
    assert state["recent_files"] == []


def test_successful_workspace_reads_and_writes_refresh_recent_files(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "existing.py").write_text("print('ok')\n", encoding="utf-8")
    kernel = Kernel()
    kernel.workspace_manager.update_project(path=str(workspace))
    reader = ReadWorkspaceFileTool(kernel.workspace_manager)
    writer = WriteWorkspaceFileTool(kernel.workspace_manager)

    read = reader.execute(
        ToolRequest(
            tool="filesystem.read_file",
            arguments={"path": "existing.py"},
            request_id="read",
        )
    )
    write = writer.execute(
        ToolRequest(
            tool="filesystem.write_file",
            arguments={"path": "new.py", "content": "print('new')\n"},
            request_id="write",
        )
    )
    denied = reader.execute(
        ToolRequest(
            tool="filesystem.read_file",
            arguments={"path": ".env"},
            request_id="secret",
        )
    )

    assert read.success
    assert write.success
    assert not denied.success
    assert kernel.workspace_manager.snapshot()["recent_files"] == [
        "new.py",
        "existing.py",
    ]