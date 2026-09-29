from pathlib import Path

from core.contracts import ToolRequest
from core.tools.vision_locate import VisionLocateTool


class FakeVisionEngine:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def inspect(self, image_path, prompt, *, json_mode, system_prompt, response_schema=None):
        self.calls.append(
            {
                "image_path": str(image_path),
                "prompt": prompt,
                "json_mode": json_mode,
                "system_prompt": system_prompt,
                "response_schema": response_schema,
            }
        )
        return self.payload


def make_request(target="Create App"):
    return ToolRequest(
        tool="vision.locate",
        arguments={"target": target},
        request_id="test-locate",
    )


def make_png(path: Path, width=1000, height=800):
    import struct
    import zlib

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def test_locate_returns_grounded_image_and_screen_coordinates(tmp_path):
    image = tmp_path / "screen.png"
    make_png(image)

    vision = FakeVisionEngine(
        {
            "model": "fake-vl",
            "json": {
                "found": True,
                "element": "Create App",
                "bbox": [700, 475, 820, 525],
                "confidence": 0.92,
            },
            "text": "",
            "ttft": 0.01,
            "request_time": 0.02,
            "output_tokens": 20,
            "tokens_per_second": 1000,
        }
    )

    tool = VisionLocateTool(
        vision,
        capture=lambda: {
            "path": str(image),
            "width": 1000,
            "height": 800,
            "origin_x": -1920,
            "origin_y": 0,
        },
    )

    result = tool.execute(make_request())

    assert result.success is True
    assert result.output["center"] == {"x": 760, "y": 400}
    assert result.output["screen_center"] == {"x": -1160, "y": 400}
    assert result.output["bbox"] == {
        "x": 700,
        "y": 380,
        "width": 120,
        "height": 40,
    }
    assert vision.calls[0]["json_mode"] is True
    system_prompt = vision.calls[0]["system_prompt"]
    assert "desktop pixels" in system_prompt
    assert "normalized 0-1000" in system_prompt
    assert "Do not copy coordinates" in system_prompt
    assert "target description is semantic" in system_prompt
    assert "search-field description may match" in system_prompt
    assert "full clickable control" in system_prompt
    assert "matches this description" in vision.calls[0]["prompt"]
    schema = vision.calls[0]["response_schema"]
    assert schema["properties"]["bbox"]["anyOf"][1] == {"type": "null"}
    assert schema["properties"]["bbox"]["anyOf"][0]["type"] == "array"
    assert schema["properties"]["bbox"]["anyOf"][0]["minItems"] == 4
    assert schema["properties"]["bbox"]["anyOf"][0]["maxItems"] == 4
    assert schema["required"] == ["found", "element", "bbox", "confidence"]
    assert schema["additionalProperties"] is False


def test_locate_rejects_low_confidence(tmp_path):
    image = tmp_path / "screen.png"
    make_png(image)

    tool = VisionLocateTool(
        FakeVisionEngine(
            {
                "json": {
                    "found": True,
                    "bbox": [100, 100, 300, 300],
                    "confidence": 0.40,
                }
            }
        ),
        capture=lambda: {
            "path": str(image),
            "width": 100,
            "height": 100,
        },
    )

    result = tool.execute(make_request("button"))

    assert result.success is False
    assert "below" in result.error


def test_locate_rejects_out_of_bounds_boxes(tmp_path):
    image = tmp_path / "screen.png"
    make_png(image, width=100, height=100)

    tool = VisionLocateTool(
        FakeVisionEngine(
            {
                "json": {
                    "found": True,
                    "bbox": [900, 100, 1100, 200],
                    "confidence": 0.95,
                }
            }
        ),
        capture=lambda: {
            "path": str(image),
            "width": 100,
            "height": 100,
        },
    )

    result = tool.execute(make_request("button"))

    assert result.success is False
    assert "outside" in result.error


def test_locate_recovers_truncated_json(tmp_path):
    image = tmp_path / "screen.png"
    make_png(image, width=1000, height=1000)

    tool = VisionLocateTool(
        FakeVisionEngine(
            {
                "model": "fake-vl",
                "json": None,
                "text": (
                    '{"found": true, "element": "Search", '
                    '"bbox": [100, 100, 300, 300], "confidence": 0.95, '
                    '"note": "truncated'
                ),
                "output_tokens": 64,
            }
        ),
        capture=lambda: {
            "path": str(image),
            "width": 1000,
            "height": 1000,
        },
    )

    result = tool.execute(make_request("Search"))

    assert result.success is True
    assert result.output["center"] == {"x": 200, "y": 200}
    assert result.output["bbox"] == {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 200,
    }
    assert result.output["json_recovered"] is True


def test_locate_not_found_is_observable(tmp_path):
    image = tmp_path / "screen.png"
    make_png(image)

    tool = VisionLocateTool(
        FakeVisionEngine(
            {
                "json": {
                    "found": False,
                    "bbox": None,
                    "confidence": 0.10,
                    "summary": "Target is not visible.",
                }
            }
        ),
        capture=lambda: {
            "path": str(image),
            "width": 100,
            "height": 100,
        },
    )

    result = tool.execute(make_request("missing"))

    assert result.success is False
    assert result.output["found"] is False
    assert result.error == "Target not found: missing"
