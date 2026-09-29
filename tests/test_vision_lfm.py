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


def test_lfm_engine_default_verification_prompt_is_target_neutral(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png")

    session = FakeSession()
    engine = LFM25VLEngine(
        base_url="http://127.0.0.1:8090/v1",
        session=session,
    )

    engine.inspect(image, "Verify that Spotify playback is active.", json_mode=True)

    system_prompt = session.post_payload["messages"][0]["content"]
    assert "Calculator window is visible" not in system_prompt
    assert "Do not rely on prior turns" in system_prompt
    assert "different application" in system_prompt
    assert "Ignore A.S.T.A.'s HUD" in system_prompt
    assert "merely repeats the requested command" in system_prompt

def test_vision_inspect_tool_returns_verification_evidence(tmp_path, capsys):
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
    telemetry = capsys.readouterr().out
    assert "[Vision] Inspect:" in telemetry
    assert "model=lfm2.5-vl-test" in telemetry
    assert "ttft=0.020s" in telemetry
    assert "request=0.040s" in telemetry
    assert "visual_match=true" in telemetry
    assert "confidence=0.95" in telemetry
    assert "verified=true" in telemetry

def test_vision_inspect_tool_accepts_capture_metadata_dict(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png")

    tool = VisionInspectTool(
        FakeVisionEngine(),
        capture=lambda: {"path": str(image), "width": 1, "height": 1},
    )

    result = tool.execute(
        ToolRequest(
            tool="vision.inspect",
            arguments={"prompt": "Is Calculator visible?"},
            request_id="test-dict-capture",
        )
    )

    assert result.success is True
    assert result.output["path"] == str(image.resolve())
    assert result.output["verified"] is True


def test_vision_inspect_tool_rejects_capture_metadata_without_path(tmp_path):
    tool = VisionInspectTool(
        FakeVisionEngine(),
        capture=lambda: {"width": 1, "height": 1},
    )

    result = tool.execute(
        ToolRequest(
            tool="vision.inspect",
            arguments={"prompt": "Is Calculator visible?"},
            request_id="test-invalid-capture",
        )
    )

    assert result.success is False
    assert "without a 'path'" in result.error

def test_lfm_engine_recovers_truncated_verification_json():
    truncated = (
        '{"visual_match":true,"confidence":0.95,'
        '"summary":"The calculator window is visible.",'
        '"observations":["calculator"]'
    )

    recovered = LFM25VLEngine._recover_truncated_json(truncated)

    assert recovered == {
        "visual_match": True,
        "confidence": 0.95,
        "summary": "The calculator window is visible.",
    }



def test_lfm_engine_uses_supplied_response_schema(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png")

    session = FakeSession()
    engine = LFM25VLEngine(
        base_url="http://127.0.0.1:8090/v1",
        session=session,
    )
    schema = {
        "type": "object",
        "properties": {"found": {"type": "boolean"}},
        "required": ["found"],
    }

    engine.inspect(
        image,
        "Locate the Search field.",
        json_mode=True,
        response_schema=schema,
    )

    assert session.post_payload["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }



def test_vision_inspect_rejects_playback_verification_without_requested_title(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake-png")

    class MismatchPlaybackEngine:
        def inspect(self, image_path, prompt, *, json_mode=True):
            assert prompt == (
                "Verify that 'hanuman chalisa' is actually playing in spotify."
            )
            return {
                "model": "lfm2.5-vl-test",
                "text": '{"visual_match":true,"confidence":1.0,"summary":"The current song is On Repeat."}',
                "json": {
                    "visual_match": True,
                    "confidence": 1.0,
                    "summary": "The current song is On Repeat.",
                },
            }

    tool = VisionInspectTool(
        MismatchPlaybackEngine(),
        capture=lambda: image,
    )

    result = tool.execute(
        ToolRequest(
            tool="vision.inspect",
            arguments={
                "prompt": (
                    "Verify that 'hanuman chalisa' is actually playing in spotify."
                )
            },
            request_id="test-playback-mismatch",
        )
    )

    assert result.success is True
    assert result.output["visual_match"] is False
    assert result.output["verified"] is False
    assert result.output["required_playback_text"] == "hanuman chalisa"
    assert result.output["verification_guard"]
