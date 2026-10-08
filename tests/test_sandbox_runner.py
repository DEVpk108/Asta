from __future__ import annotations

import sys
from pathlib import Path

from core import Kernel
from core.contracts import ToolRequest
from core.tools import RunCommandTool
from core.tools.sandbox import (
    SandboxProcessResult,
    _bounded_process,
    _resolve_runtime_executable,
)


class FakeRuntime:
    runtime_name = "docker"

    def __init__(self, *, start_result=None, create_result=None):
        self.calls = []
        self.start_result = start_result or SandboxProcessResult(0, stdout="ASTA_OK\n")
        self.create_result = create_result or SandboxProcessResult(0, stdout="container-id\n")
        self.snapshot_at_create = None
        self.snapshot_files = {}

    def run(self, arguments, *, timeout, output_limit):
        args = list(arguments)
        self.calls.append((args, timeout, output_limit))
        if args[0] == "create":
            mount = args[args.index("--mount") + 1]
            source = mount.split("source=", 1)[1].split(",target=", 1)[0]
            self.snapshot_at_create = Path(source)
            self.snapshot_files = {
                path.relative_to(self.snapshot_at_create).as_posix(): path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                for path in self.snapshot_at_create.rglob("*")
                if path.is_file()
            }
            return self.create_result
        if args[0] == "start":
            return self.start_result
        return SandboxProcessResult(0)


def _request(target, arguments=None, *, cwd=".", timeout=30.0):
    return ToolRequest(
        tool="system.run_command",
        arguments={"target": target, "arguments": arguments or [], "cwd": cwd},
        request_id="sandbox-test",
        timeout_seconds=timeout,
    )


def _tool(tmp_path, runtime):
    workspace = tmp_path / "project"
    workspace.mkdir()
    kernel = Kernel()
    kernel.workspace_manager.update_project(path=str(workspace))
    return workspace, RunCommandTool(kernel.workspace_manager, runtime=runtime)


def test_runs_python_only_in_a_sanitized_readonly_networkless_snapshot(tmp_path):
    runtime = FakeRuntime()
    workspace, tool = _tool(tmp_path, runtime)
    (workspace / "main.py").write_text('print("ASTA_OK")\n', encoding="utf-8")
    (workspace / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (workspace / ".env.example").write_text("TEMPLATE=value\n", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("private metadata\n", encoding="utf-8")
    (workspace / ".venv").mkdir()
    (workspace / ".venv" / "marker").write_text("dependency data\n", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside data\n", encoding="utf-8")
    try:
        (workspace / "outside-link.txt").symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pass

    result = tool.execute(_request(sys.executable, ["main.py"]))

    assert result.success
    assert result.output["returncode"] == 0
    assert result.output["stdout"] == "ASTA_OK\n"
    assert result.output["sandbox"] == {
        "runtime": "docker",
        "network": "none",
        "workspace_snapshot": "read-only",
        "output_truncated": False,
    }
    assert tool.definition.risk_level == "critical"
    create = runtime.calls[0][0]
    assert create[0] == "create"
    assert "--pull=never" in create
    assert "--network=none" in create
    assert "--read-only" in create
    assert "--cap-drop=ALL" in create
    assert "--security-opt=no-new-privileges" in create
    assert "--memory=1g" in create
    assert "--memory-swap=1g" in create
    assert "--pids-limit=64" in create
    assert create[create.index("--mount") + 1].endswith(",readonly")
    assert create[-2:] == ["asta-python-sandbox:1", "main.py"]
    snapshot = runtime.snapshot_at_create
    assert snapshot is not None
    assert "main.py" in runtime.snapshot_files
    assert ".env.example" in runtime.snapshot_files
    assert ".env" not in runtime.snapshot_files
    assert "outside-link.txt" not in runtime.snapshot_files
    assert not any(path.startswith(".git/") for path in runtime.snapshot_files)
    assert not any(path.startswith(".venv/") for path in runtime.snapshot_files)
    assert not snapshot.exists()
    assert runtime.calls[-1][0] == ["rm", "--force", runtime.calls[0][0][runtime.calls[0][0].index("--name") + 1]]


def test_rejects_unsupported_target_and_workspace_escape_before_runtime(tmp_path):
    runtime = FakeRuntime()
    workspace, tool = _tool(tmp_path, runtime)

    unsupported = tool.execute(_request("bash", ["-c", "echo unsafe"]))
    escaped = tool.execute(_request("python", ["main.py"], cwd="../"))

    assert not unsupported.success
    assert "Python and pytest" in unsupported.error
    assert not escaped.success
    assert "traversal" in escaped.error.lower()
    assert runtime.calls == []


def test_requires_an_active_workspace_and_never_runs_on_host():
    runtime = FakeRuntime()
    kernel = Kernel()
    tool = RunCommandTool(kernel.workspace_manager, runtime=runtime)

    result = tool.execute(_request("python", ["-c", "print('unsafe')"]))

    assert not result.success
    assert "No active project workspace" in result.error
    assert runtime.calls == []


def test_timeout_force_removes_container_and_never_falls_back(tmp_path):
    runtime = FakeRuntime(
        start_result=SandboxProcessResult(137, stderr="timeout", timed_out=True)
    )
    workspace, tool = _tool(tmp_path, runtime)
    (workspace / "main.py").write_text("pass\n", encoding="utf-8")

    result = tool.execute(_request("python", ["main.py"], timeout=0.05))

    assert not result.success
    assert "timed out" in result.error.lower()
    assert any(call[0][:2] == ["kill", runtime.calls[0][0][runtime.calls[0][0].index("--name") + 1]] for call in runtime.calls)
    assert runtime.calls[-1][0][0:2] == ["rm", "--force"]


def test_fails_closed_when_local_image_cannot_be_created(tmp_path):
    runtime = FakeRuntime(create_result=SandboxProcessResult(1, stderr="image missing"))
    workspace, tool = _tool(tmp_path, runtime)
    (workspace / "main.py").write_text("pass\n", encoding="utf-8")

    result = tool.execute(_request("python", ["main.py"]))

    assert not result.success
    assert "never pulls images" in result.error
    assert all(call[0][0] != "start" for call in runtime.calls)


def test_runtime_output_is_bounded_and_marked(tmp_path):
    runtime = FakeRuntime(
        start_result=SandboxProcessResult(
            0,
            stdout="x" * (128 * 1024),
            stderr="y" * (128 * 1024),
            stdout_truncated=True,
            stderr_truncated=True,
        )
    )
    workspace, tool = _tool(tmp_path, runtime)
    (workspace / "main.py").write_text("pass\n", encoding="utf-8")

    result = tool.execute(_request("python", ["main.py"]))

    assert result.success
    assert result.output["sandbox"]["output_truncated"] is True
    assert "stdout truncated" in result.output["stdout"]
    assert "stderr truncated" in result.output["stderr"]


def test_runtime_adapter_bounds_output_and_kills_timed_out_cli():
    noisy = _bounded_process(
        [sys.executable, "-c", "print('x' * 100000)"],
        timeout=5,
        output_limit=32,
    )
    sleeper = _bounded_process(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=0.05,
        output_limit=32,
    )

    assert noisy.returncode == 0
    assert len(noisy.stdout.encode("utf-8")) <= 32
    assert noisy.stdout_truncated
    assert sleeper.timed_out


def test_windows_runtime_resolution_chooses_native_exe_not_shell_shim(monkeypatch):
    requested = []

    def which(candidate):
        requested.append(candidate)
        if candidate == "docker.exe":
            return r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
        if candidate == "docker":
            return r"C:\tools\docker.cmd"
        return None

    monkeypatch.setattr("core.tools.sandbox.shutil.which", which)

    resolved = _resolve_runtime_executable("docker", is_windows=True)

    assert resolved == r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    assert requested == ["docker.exe"]


def test_windows_runtime_resolution_rejects_only_cmd_shim(monkeypatch):
    monkeypatch.setattr(
        "core.tools.sandbox.shutil.which",
        lambda candidate: r"C:\tools\docker.cmd" if candidate == "docker.exe" else None,
    )

    assert _resolve_runtime_executable("docker", is_windows=True) is None