from pathlib import Path

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
    assert kill_calls[0][0] == ["taskkill", "/PID", "4242", "/T", "/F"]


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
