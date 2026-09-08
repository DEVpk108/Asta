from types import SimpleNamespace

from voice.recognition_engine import RecognitionEngine


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return [SimpleNamespace(text="namaste ASTA")], SimpleNamespace(
            language="hi",
            language_probability=0.97,
        )


def test_recognition_does_not_force_english(monkeypatch):
    fake_model = FakeWhisper()
    engine = RecognitionEngine.__new__(RecognitionEngine)
    engine.model = fake_model
    engine.model_name = "medium"
    engine.beam_size = 5
    engine.language = None
    engine.debug = False

    text = engine.transcribe("audio")

    assert text == "namaste ASTA"
    assert fake_model.calls[0][1]["language"] is None
    assert "Hindi" in fake_model.calls[0][1]["initial_prompt"]
    assert "Hinglish" in fake_model.calls[0][1]["initial_prompt"]
