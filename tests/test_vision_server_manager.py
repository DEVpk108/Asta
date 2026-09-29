from pathlib import Path

from vision.vision_server_manager import VisionServerManager


def test_discover_model_paths_prefers_lfm25_vl(tmp_path):
    model = tmp_path / "LFM2.5-VL-1.6B-Q8_0.gguf"
    mmproj = tmp_path / "mmproj-LFM2.5-VL-1.6b-F16.gguf"
    other = tmp_path / "Nemotron-Nano-4B.gguf"
    model.write_bytes(b"")
    mmproj.write_bytes(b"")
    other.write_bytes(b"")

    found_model, found_mmproj = VisionServerManager.discover_model_paths(tmp_path)

    assert found_model == model
    assert found_mmproj == mmproj


def test_discover_model_paths_prefers_lfm25_vl_3b_for_grounding(tmp_path):
    model_16 = tmp_path / "LFM2.5-VL-1.6B-Q8_0.gguf"
    model_3 = tmp_path / "LFM2.5-VL-3B-Q4_K_M.gguf"
    mmproj_16 = tmp_path / "mmproj-LFM2.5-VL-1.6b-F16.gguf"
    mmproj_3 = tmp_path / "mmproj-LFM2.5-VL-3B-F16.gguf"
    for path in (model_16, model_3, mmproj_16, mmproj_3):
        path.write_bytes(b"")

    found_model, found_mmproj = VisionServerManager.discover_model_paths(tmp_path)

    assert found_model == model_3
    assert found_mmproj == mmproj_3


def test_build_command_uses_local_model_and_mmproj(tmp_path):
    manager = VisionServerManager(
        base_url="http://127.0.0.1:8090/v1",
        server_path=str(tmp_path / "llama-server.exe"),
        context_size=8192,
        gpu_layers=99,
        jinja=True,
        mmproj_offload=True,
    )

    command = manager._build_command(
        "E:\\Projects\\llama\\llama-server.exe",
        "E:\\Projects\\llama\\models\\LFM2.5-VL-1.6B-Q8_0.gguf",
        "E:\\Projects\\llama\\models\\mmproj-LFM2.5-VL-1.6b-F16.gguf",
    )

    assert command[:6] == [
        "E:\\Projects\\llama\\llama-server.exe",
        "-m",
        "E:\\Projects\\llama\\models\\LFM2.5-VL-1.6B-Q8_0.gguf",
        "--mmproj",
        "E:\\Projects\\llama\\models\\mmproj-LFM2.5-VL-1.6b-F16.gguf",
        "--host",
    ]
    assert "--port" in command
    assert "8090" in command
    assert "-ngl" in command
    assert "99" in command
    assert "--mmproj-offload" in command


def test_vision_server_is_lazy_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTA_VISION_PRELOAD", raising=False)
    monkeypatch.setenv("ASTA_VISION_MODEL_DIR", str(tmp_path))

    manager = VisionServerManager(base_url="http://127.0.0.1:8090/v1")

    assert manager.preload is False
    assert manager.process is None
    assert manager.owned is False
