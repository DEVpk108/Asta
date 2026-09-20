from core import Kernel
from core.workspace_runtime import WorkspaceRuntimeModule


def test_workspace_runtime_populates_runtime_discoverable_state():
    kernel = Kernel()
    module = WorkspaceRuntimeModule(kernel)

    module.initialize()
    try:
        workspace = kernel.workspace_manager.snapshot()
    finally:
        module.shutdown()

    assert workspace["project_name"]
    assert workspace["project_path"]
    assert workspace["environment"]["os"]
    assert workspace["environment"]["python"]
    assert "machine" in workspace["hardware"]
