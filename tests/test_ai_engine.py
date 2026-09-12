import json

from ai.openai_engine import AIEngine


class FakeResponse:
    def __init__(self, events=None, json_data=None):
        self.events = events or []
        self.json_data = {"ok": True} if json_data is None else json_data
        self.encoding = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return self.json_data

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


def test_local_engine_uses_direct_loopback_transport():
    engine = AIEngine()
    assert engine.base_url == "http://127.0.0.1:1234"
    assert engine.session.trust_env is False


def test_warmup_loads_model_only_when_not_already_loaded(monkeypatch):
    calls = []

    def fake_get(_session, url, **kwargs):
        calls.append(("GET", url, kwargs))
        return FakeResponse(json_data={"models": []})

    def fake_post(_session, url, **kwargs):
        calls.append(("POST", url, kwargs))
        return FakeResponse()

    monkeypatch.setattr("ai.openai_engine.requests.Session.get", fake_get)
    monkeypatch.setattr("ai.openai_engine.requests.Session.post", fake_post)

    engine = AIEngine()
    assert engine.warmup() is True
    assert engine.warmed is True
    assert engine.previous_response_id is None

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v1/models")
    assert calls[1][0] == "POST"
    assert calls[1][1].endswith("/api/v1/models/load")
    assert calls[1][2]["json"] == {"model": engine.model}

    assert calls[2][0] == "POST"
    assert calls[2][1].endswith("/api/v1/chat")
    assert calls[2][2]["json"]["store"] is False
    assert calls[2][2]["json"]["stream"] is False
    assert calls[2][2]["json"]["max_output_tokens"] == 1
    assert calls[2][2]["json"]["reasoning"] == "off"


def test_warmup_reuses_existing_model_instance_without_loading_again(monkeypatch):
    calls = []

    def fake_get(_session, url, **kwargs):
        calls.append(("GET", url, kwargs))
        return FakeResponse(
            json_data={
                "models": [
                    {
                        "key": "nvidia/nemotron-3-nano-4b",
                        "loaded_instances": [
                            {
                                "id": "nvidia/nemotron-3-nano-4b",
                                "config": {"parallel": 4},
                            }
                        ],
                    }
                ]
            }
        )

    def fake_post(_session, url, **kwargs):
        calls.append(("POST", url, kwargs))
        return FakeResponse()

    monkeypatch.setattr("ai.openai_engine.requests.Session.get", fake_get)
    monkeypatch.setattr("ai.openai_engine.requests.Session.post", fake_post)

    engine = AIEngine()
    assert engine.warmup() is True
    assert engine.warmed is True
    assert engine.model_instance_id == "nvidia/nemotron-3-nano-4b"

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v1/models")
    assert len(calls) == 2
    assert calls[1][0] == "POST"
    assert calls[1][1].endswith("/api/v1/chat")


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
        lambda _session, *args, **kwargs: FakeResponse(events),
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
        lambda _session, *args, **kwargs: FakeResponse(events),
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

    def fake_post(_session, url, **kwargs):
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


def test_response_guard_stops_stream_after_three_sentences(monkeypatch):
    events = [
        {"type": "message.delta", "content": "First sentence. Second sentence. Third sentence. Fourth sentence."},
        {
            "type": "chat.end",
            "result": {
                "response_id": "guarded-response",
                "output": [{
                    "type": "message",
                    "content": "First sentence. Second sentence. Third sentence. Fourth sentence.",
                }],
                "stats": {
                    "input_tokens": 4,
                    "total_output_tokens": 20,
                    "reasoning_output_tokens": 0,
                    "tokens_per_second": 60.0,
                },
            },
        },
    ]

    monkeypatch.setattr(
        "ai.openai_engine.requests.Session.post",
        lambda _session, *args, **kwargs: FakeResponse(events),
    )

    engine = AIEngine()
    spoken = []
    result = engine.generate_response("hello", on_sentence=spoken.append)

    assert spoken == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert result == "First sentence. Second sentence. Third sentence. Fourth sentence."
    assert engine.max_sentences == 3
