from __future__ import annotations

import json
from pathlib import Path

from core.contracts import ToolRequest
from core.tools.vision_inspect import VisionInspectTool
from vision.lfm2_5_vl_engine import LFM25VLEngine


class FakeResponse:
    def __init__(self, events):
        self._events = events

    def json(self):
        return {"data": self._events}

    def raise_for_status(self):
        return None

    def iter_lines(self, chunk_size=1, decode_unicode=True):
        for event in self._events:
            yield "data: " + json.dumps(event)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self):
        self.post_payload = None
        self.trust_env = False

    def get(self, url, timeout):
        return FakeResponse(
            [{"id": "lfm2.5-vl-test"}]
        )

    def post(self, url, json, stream, timeout, headers):
        self.post_payload = json
        return FakeResponse(
            [
                {
                    "id": "vision-1",
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    '{"visual_match":true,'
                                    '"confidence":0.96,'
                                    '"summary":"A skull is visible.",'
                                    '"observations":["skull"]}'
                                )
                            }
                        }
                    ],
                },
                {
                    "timings": {
                        "predicted_n": 8,
                        "predicted_ms": 80.0,
                        "predicted_per_second": 100.0,
                    },
                    "usage": {
                        "completion_tokens": 8,
                    },
                },
                {"choices": []},
            ]
        )


def test_lfm_engine_builds_multimodal_openai_request(tmp_path):
    image = tmp_path / "skull.png"
    image.write_bytes(b"fake-png")

    session = FakeSession()
    engine = LFM25VLEngine(
        base_url="http://127.0.0.1:8090/v1",
        session=session,
    )

    result = engine.inspect(image, "What is this?", json_mode=True)

    assert result["json"]["visual_match"] is True
    assert result["json"]["confidence"] == 0.96
    assert result["tokens_per_second"] == 100.0

    content = session.post_payload["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What is this?"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert session.post_payload["response_format"] == {"type": "json_object"}


class FakeVisionEngine:
    def inspect(self, image_path, prompt, *, json_mode=True):
        assert Path(image_path).name == "screen.png"
        assert prompt == "Is Calculator visible?"
        assert json_mode is True
        return {
            "model": "lfm2.5-vl-test",
            "text": '{"visual_match":true}',
            "json": {
                "visual_match": True,
                "confidence": 0.95,
                "summary": "Calculator is visible.",
            },
            "ttft": 0.02,
            "request_time": 0.04,
            "output_tokens": 8,
            "tokens_per_second": 150.0,
        }


def test_vision_inspect_tool_returns_verification_evidence(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png")

    tool = VisionInspectTool(
        FakeVisionEngine(),
        capture=lambda: image,
    )

    result = tool.execute(
        ToolRequest(
            tool="vision.inspect",
            arguments={"prompt": "Is Calculator visible?"},
            request_id="test",
        )
    )

    assert result.success
    assert result.output["visual_match"] is True
    assert result.output["verified"] is True
    assert result.output["confidence"] == 0.95
    assert result.output["tokens_per_second"] == 150.0
