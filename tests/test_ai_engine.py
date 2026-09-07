import json

from ai.openai_engine import AIEngine


class FakeResponse:
    def __init__(self, events):
        self.events = events
        self.encoding = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, chunk_size=None, decode_unicode=True):
        assert chunk_size == 1
        assert decode_unicode is True
        for event in self.events:
            yield f"data: {json.dumps(event)}"


def test_extract_message_text_ignores_reasoning_items():
    output = [
        {"type": "reasoning", "content": "internal thought"},
        {"type": "message", "content": "Hello ASTA!"},
    ]

    assert AIEngine._extract_message_text(output) == "Hello ASTA!"


def test_reasoning_is_off_by_default_for_voice_latency():
    engine = AIEngine()
    assert engine.reasoning == "off"


def test_sse_stream_uses_low_latency_chunk_size(monkeypatch):
    events = [
        {"type": "message.delta", "content": "Hello!"},
        {
            "type": "chat.end",
            "result": {
                "response_id": "stream-response",
                "output": [{"type": "message", "content": "Hello!"}],
                "stats": {
                    "input_tokens": 4,
                    "total_output_tokens": 2,
                    "reasoning_output_tokens": 0,
                    "tokens_per_second": 60.0,
                    "time_to_first_token_seconds": 0.15,
                },
            },
        },
    ]

    monkeypatch.setattr(
        "ai.openai_engine.requests.Session.post",
        lambda *args, **kwargs: FakeResponse(events),
    )

    engine = AIEngine()
    result = engine.generate_response("hello")

    assert result == "Hello!"


def test_latency_events_are_tracked(monkeypatch):
    events = [
        {"type": "prompt_processing.start"},
        {"type": "prompt_processing.end"},
        {"type": "message.start"},
        {"type": "message.delta", "content": "Hello!"},
        {
            "type": "chat.end",
            "result": {
                "response_id": "timed-response",
                "output": [{"type": "message", "content": "Hello!"}],
                "stats": {
                    "input_tokens": 4,
                    "total_output_tokens": 2,
                    "reasoning_output_tokens": 0,
                    "tokens_per_second": 60.0,
                    "time_to_first_token_seconds": 0.12,
                },
            },
        },
    ]

    monkeypatch.setattr(
        "ai.openai_engine.requests.Session.post",
        lambda *args, **kwargs: FakeResponse(events),
    )

    engine = AIEngine()
    attempt = engine._request("hello", None, 256)

    assert attempt["prompt_processing_time"] is not None
    assert attempt["message_start_to_first_delta"] is not None
    assert attempt["ttft"] == 0.12


def test_reasoning_exhaustion_retries_with_larger_budget(monkeypatch):
    calls = []

    first_events = [
        {
            "type": "chat.end",
            "result": {
                "response_id": "failed-response",
                "output": [{"type": "reasoning", "content": "still thinking"}],
                "stats": {
                    "input_tokens": 10,
                    "total_output_tokens": 4,
                    "reasoning_output_tokens": 4,
                    "tokens_per_second": 50.0,
                },
            },
        }
    ]
    second_events = [
        {
            "type": "chat.end",
            "result": {
                "response_id": "successful-response",
                "output": [{"type": "message", "content": "I am ready."}],
                "stats": {
                    "input_tokens": 10,
                    "total_output_tokens": 8,
                    "reasoning_output_tokens": 4,
                    "tokens_per_second": 45.0,
                },
            },
        }
    ]

    responses = [FakeResponse(first_events), FakeResponse(second_events)]

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("ai.openai_engine.requests.Session.post", fake_post)

    engine = AIEngine(max_output_tokens=4, reasoning_retry_tokens=8, reasoning="on")
    spoken = []
    result = engine.generate_response("hello", on_sentence=spoken.append)

    assert result == "I am ready."
    assert spoken == ["I am ready."]
    assert len(calls) == 2
    assert calls[0][1]["json"]["max_output_tokens"] == 4
    assert calls[1][1]["json"]["max_output_tokens"] == 8
    assert calls[0][1]["json"]["reasoning"] == "on"
    assert engine.previous_response_id == "successful-response"
