from core import Kernel, WorkspaceManager


def test_workspace_manager_updates_and_snapshots_state():
    kernel = Kernel()
    manager = WorkspaceManager(event_bus=kernel.event_bus)
    events = []
    kernel.event_bus.subscribe(
        "workspace_updated",
        lambda workspace: events.append(workspace),
    )

    manager.update_project(
        name="A.S.T.A.",
        path="C:/Asta",
        repository="https://github.com/DEVpk108/Asta.git",
        branch="feat/asta-hud",
    )
    manager.set_active_files(["core/workspace_manager.py", "core/context_builder.py"])
    manager.add_recent_file("core/workspace_manager.py")
    manager.add_recent_file("core/workspace_manager.py")
    manager.set_environment({"python": "3.12"})
    manager.set_hardware({"machine": "x86_64"})

    snapshot = manager.snapshot()

    assert snapshot["project_name"] == "A.S.T.A."
    assert snapshot["repository"].endswith("DEVpk108/Asta.git")
    assert snapshot["branch"] == "feat/asta-hud"
    assert snapshot["active_files"] == [
        "core/workspace_manager.py",
        "core/context_builder.py",
    ]
    assert snapshot["recent_files"] == ["core/workspace_manager.py"]
    assert snapshot["environment"] == {"python": "3.12"}
    assert snapshot["hardware"] == {"machine": "x86_64"}
    assert snapshot["updated_at"] > 0
    assert events
    assert events[-1]["hardware"] == {"machine": "x86_64"}


def test_kernel_owns_workspace_manager_and_returns_detached_snapshot():
    kernel = Kernel()
    kernel.workspace_manager.update_project(name="A.S.T.A.")

    workspace = kernel.workspace
    workspace["project_name"] = "changed locally"

    assert kernel.workspace_manager.snapshot()["project_name"] == "A.S.T.A."
