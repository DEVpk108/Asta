from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

from ai.llama_server_manager import LlamaServerManager


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeProcess:
    def __init__(self):
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

def test_manager_starts_server_with_configured_model_and_waits_until_ready(
    monkeypatch,
    tmp_path,
):
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    server.write_text("")
    model.write_text("")

    readiness = iter([False, True])
    popen_calls = []
    kill_calls = []

    def fake_get(_session, url, **kwargs):
        assert url.endswith("/v1/models")
        return FakeResponse()

    def fake_server_is_ready(self):
        return next(readiness)

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeProcess()

    def fake_run(command, **kwargs):
        kill_calls.append((command, kwargs))
        return None

    monkeypatch.setattr(
        "ai.llama_server_manager.LlamaServerManager._server_is_ready",
        fake_server_is_ready,
    )
    monkeypatch.setattr(
        "ai.llama_server_manager.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "ai.llama_server_manager.subprocess.run",
        fake_run,
    )

    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8080/v1",
        server_path=str(server),
        model_path=str(model),
        context_size=8192,
        gpu_layers=99,
        jinja=True,
        reasoning="off",
        startup_timeout=1,
    )

    assert manager.ensure_running() is True
    assert manager.owned is True
    assert manager.running is True

    command = popen_calls[0][0]
    assert command[:7] == [
        str(server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    assert "-c" in command
    assert "8192" in command
    assert "-ngl" in command
    assert "99" in command
    assert "--jinja" in command
    assert command[-2:] == ["--reasoning", "off"]

    manager.stop()

    assert manager.owned is False
    assert manager.process is None
    if os.name == "nt":
        assert kill_calls[0][0] == ["taskkill", "/PID", "4242", "/T", "/F"]
    else:
        assert not kill_calls


def test_stop_waits_for_windows_child_and_retains_ownership_on_failure(monkeypatch):
    manager = LlamaServerManager(base_url="http://127.0.0.1:8080/v1")
    process = FakeProcess()
    calls = []
    manager.process = process
    manager.owned = True
    monkeypatch.setattr("ai.llama_server_manager.os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        "ai.llama_server_manager.subprocess.run",
        lambda *args, **kwargs: calls.append(args[0]),
    )
    def never_exits(timeout=None):
        raise subprocess.TimeoutExpired("llama-server", timeout)
    def cannot_kill():
        raise OSError("permission denied")
    process.wait = never_exits
    process.kill = cannot_kill
    assert manager.stop() is False
    assert manager.owned and manager.process is process
    assert calls[0] == ["taskkill", "/PID", "4242", "/T", "/F"]


def test_coding_namespace_never_discovers_general_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTA_LLM_MODEL_PATH", str(tmp_path / "general.gguf"))
    server = tmp_path / "llama-server.exe"
    server.write_text("")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "general.gguf").write_text("")
    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8081/v1", server_path=str(server),
        env_prefix="ASTA_CODING_LLM",
    )
    assert not manager.model_path
    assert manager._resolve_model_path(str(server)) is None
    assert manager.context_size == 4096
    assert manager.batch_size == 256
    assert manager.ubatch_size == 128


def test_manager_does_not_own_already_running_server(monkeypatch):
    monkeypatch.setattr(
        "ai.llama_server_manager.LlamaServerManager._server_is_ready",
        lambda self: True,
    )

    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8080/v1",
        server_path="missing-server",
        model_path="missing-model.gguf",
    )

    assert manager.ensure_running() is True
    assert manager.owned is False
    assert manager.process is None


def test_manager_can_discover_single_gguf_next_to_server(tmp_path):
    server_dir = Path(tmp_path)
    server = server_dir / "llama-server.exe"
    model_dir = server_dir / "models"
    model = model_dir / "asta.gguf"

    server.write_text("")
    model_dir.mkdir()
    model.write_text("")

    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8080/v1",
        server_path=str(server),
        model_path=None,
    )

    assert manager._resolve_model_path(str(server)) == str(model)


def test_manager_requires_explicit_model_when_multiple_ggufs(tmp_path):
    server_dir = Path(tmp_path)
    server = server_dir / "llama-server.exe"
    model_dir = server_dir / "models"

    server.write_text("")
    model_dir.mkdir()
    (model_dir / "a.gguf").write_text("")
    (model_dir / "b.gguf").write_text("")

    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8080/v1",
        server_path=str(server),
    )

    assert manager._resolve_model_path(str(server)) is None

def test_manager_emits_memory_friendly_server_tuning_flags(monkeypatch, tmp_path):
    server = tmp_path / "llama-server.exe"
    model = tmp_path / "model.gguf"
    server.write_text("")
    model.write_text("")

    monkeypatch.setenv("ASTA_LLM_BATCH_SIZE", "1024")
    monkeypatch.setenv("ASTA_LLM_UBATCH_SIZE", "256")
    monkeypatch.setenv("ASTA_LLM_PARALLEL", "1")
    monkeypatch.setenv("ASTA_LLM_FLASH_ATTN", "on")
    monkeypatch.setenv("ASTA_LLM_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("ASTA_LLM_CACHE_TYPE_V", "q8_0")

    manager = LlamaServerManager(
        base_url="http://127.0.0.1:8080/v1",
        server_path=str(server),
        model_path=str(model),
        context_size=4096,
        gpu_layers=99,
    )

    command = manager._build_command(str(server), str(model))

    assert "-b" in command and "1024" in command
    assert "-ub" in command and "256" in command
    assert "-np" in command and "1" in command
    assert "-fa" in command and "on" in command
    assert "-ctk" in command and "q8_0" in command
    assert "-ctv" in command and "q8_0" in command



def test_unready_owned_process_is_not_replaced_after_failed_stop(monkeypatch):
    manager = LlamaServerManager(base_url="http://127.0.0.1:8080/v1")
    process = FakeProcess()
    manager.process = process
    manager.owned = True
    monkeypatch.setattr(manager, "_server_is_ready", lambda: False)
    monkeypatch.setattr(manager, "_wait_until_ready", lambda: False)
    monkeypatch.setattr(manager, "stop", lambda: False)
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("must retain the existing child handle")
    monkeypatch.setattr("ai.llama_server_manager.subprocess.Popen", forbidden_spawn)
    assert not manager.ensure_running()
    assert manager.process is process and manager.owned


def test_dead_child_does_not_claim_ownership_of_replacement_endpoint(monkeypatch):
    manager = LlamaServerManager(base_url="http://127.0.0.1:8080/v1")
    process = FakeProcess()
    process.returncode = 0
    manager.process = process
    manager.owned = True
    monkeypatch.setattr(manager, "_server_is_ready", lambda: True)
    assert manager.ensure_running()
    assert manager.process is None and not manager.owned
