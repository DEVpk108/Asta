import json

from ai.openai_engine import AIEngine


class FakeResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        for event in self.events:
            yield f"data: {json.dumps(event)}"


def test_extract_message_text_ignores_reasoning_items():
    output = [
        {"type": "reasoning", "content": "internal thought"},
        {"type": "message", "content": "Hello ASTA!"},
    ]

    assert AIEngine._extract_message_text(output) == "Hello ASTA!"


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

    monkeypatch.setattr("ai.openai_engine.requests.post", fake_post)

    engine = AIEngine(max_output_tokens=4, reasoning_retry_tokens=8)
    spoken = []
    result = engine.generate_response("hello", on_sentence=spoken.append)

    assert result == "I am ready."
    assert spoken == ["I am ready."]
    assert len(calls) == 2
    assert calls[0][1]["json"]["max_output_tokens"] == 4
    assert calls[1][1]["json"]["max_output_tokens"] == 8
    assert engine.previous_response_id == "successful-response"
