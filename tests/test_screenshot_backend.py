import struct

from vision.screenshot_backend import _encode_rgb_png


def test_encode_rgb_png_produces_valid_png_signature_and_header():
    # One opaque blue pixel in BGRA order.
    png = _encode_rgb_png(1, 1, bytes([255, 0, 0, 255]))

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert png[12:16] == b"IHDR"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (1, 1)


def test_capture_screenshot_uses_windows_backend(monkeypatch, tmp_path):
    import vision.screenshot_backend as backend

    called = {}

    def fake_capture(path):
        called["path"] = path
        return {"path": str(path), "platform": "windows"}

    monkeypatch.setattr(backend.platform, "system", lambda: "Windows")
    monkeypatch.setattr(backend, "_capture_windows", fake_capture)

    result = backend.capture_screenshot(tmp_path)

    assert result["platform"] == "windows"
    assert called["path"].parent == tmp_path
    assert called["path"].suffix == ".png"
