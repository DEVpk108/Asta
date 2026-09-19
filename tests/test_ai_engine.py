import json

from ai.llama_cpp_engine import LlamaCppEngine


class FakeResponse:
    def __init__(self, events=None, json_data=None):
        self.events = events or []
        self.json_data = {} if json_data is None else json_data
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
            if isinstance(event, str):
                yield event
            else:
                yield f"data: {json.dumps(event)}"


def _models_response():
    return FakeResponse(json_data={
        "object": "list",
        "data": [{"id": "asta-local"}],
    })


def test_llama_cpp_engine_uses_direct_loopback_transport():
    engine = LlamaCppEngine()
    assert engine.base_url == "http://127.0.0.1:8080/v1"
    assert engine.session.trust_env is False


def test_warmup_discovers_model_and_calls_chat_completions(monkeypatch):
    calls = []

    def fake_get(_session, url, **kwargs):
        calls.append(("GET", url, kwargs))
        return _models_response()

    def fake_post(_session, url, **kwargs):
        calls.append(("POST", url, kwargs))
        return FakeResponse(json_data={
            "id": "warmup",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        })

    monkeypatch.setattr("ai.llama_cpp_engine.requests.Session.get", fake_get)
    monkeypatch.setattr("ai.llama_cpp_engine.requests.Session.post", fake_post)

    engine = LlamaCppEngine()
    assert engine.warmup() is True
    assert engine.warmed is True
    assert engine.model == "asta-local"

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/v1/models")
    assert calls[1][0] == "POST"
    assert calls[1][1].endswith("/v1/chat/completions")
    payload = calls[1][2]["json"]
    assert payload["model"] == "asta-local"
    assert payload["messages"][0]["role"] == "system"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 1


def test_sse_stream_uses_openai_chat_completion_shape(monkeypatch):
    events = [
        {
            "id": "stream-response",
            "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        },
        {
            "id": "stream-response",
            "choices": [{"delta": {"content": "ASTA!"}, "finish_reason": None}],
        },
        {
            "id": "stream-response",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
            "timings": {
                "prompt_n": 4,
                "predicted_n": 2,
                "predicted_per_second": 42.0,
            },
        },
        "[DONE]",
    ]

    monkeypatch.setattr(
        "ai.llama_cpp_engine.requests.Session.post",
        lambda _session, *args, **kwargs: FakeResponse(events),
    )

    engine = LlamaCppEngine(model="asta-local")
    spoken = []
    result = engine.generate_response("hello", on_sentence=spoken.append)

    assert result == "Hello ASTA!"
    assert spoken == ["Hello ASTA!"]


def test_conversation_history_is_sent_on_subsequent_requests(monkeypatch):
    events = [
        FakeResponse([
            {
                "id": "one",
                "choices": [{"delta": {"content": "First reply."}}],
            },
            {
                "id": "one",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "timings": {},
            },
            "[DONE]",
        ]),
        FakeResponse([
            {
                "id": "two",
                "choices": [{"delta": {"content": "Second reply."}}],
            },
            {
                "id": "two",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
                "timings": {},
            },
            "[DONE]",
        ]),
    ]
    calls = []

    def fake_post(_session, url, **kwargs):
        calls.append(kwargs["json"])
        return events.pop(0)

    monkeypatch.setattr(
        "ai.llama_cpp_engine.requests.Session.post",
        fake_post,
    )

    engine = LlamaCppEngine(model="asta-local")
    assert engine.generate_response("hello") == "First reply."
    assert engine.generate_response("follow up") == "Second reply."

    assert len(calls) == 2
    assert calls[1]["messages"][-2:] == [
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "follow up"},
    ]


def test_system_prompt_reset_clears_conversation():
    engine = LlamaCppEngine(model="asta-local")
    engine._messages.append({"role": "user", "content": "old context"})

    engine.set_system_prompt("New system prompt.")

    assert engine.system_prompt == "New system prompt."
    assert engine._messages == [
        {"role": "system", "content": "New system prompt."}
    ]


def test_reasoning_exhaustion_retries_with_larger_budget(monkeypatch):
    responses = [
        FakeResponse([
            {
                "id": "failed",
                "choices": [{"delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "timings": {},
            },
            "[DONE]",
        ]),
        FakeResponse([
            {
                "id": "successful",
                "choices": [{"delta": {"content": "I am ready."}}],
            },
            {
                "id": "successful",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "timings": {"predicted_per_second": 45.0},
            },
            "[DONE]",
        ]),
    ]
    calls = []

    def fake_post(_session, url, **kwargs):
        calls.append(kwargs["json"])
        return responses.pop(0)

    monkeypatch.setattr(
        "ai.llama_cpp_engine.requests.Session.post",
        fake_post,
    )

    engine = LlamaCppEngine(
        model="asta-local",
        max_output_tokens=4,
        reasoning_retry_tokens=8,
    )
    spoken = []
    assert engine.generate_response("hello", on_sentence=spoken.append) == "I am ready."
    assert spoken == ["I am ready."]
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 4
    assert calls[1]["max_tokens"] == 8
