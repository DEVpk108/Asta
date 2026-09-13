from types import SimpleNamespace

from voice.recognition_engine import RecognitionEngine
from voice.stt_language_router import STTLanguageRouter


class FakeWhisper:
    def __init__(self, language="hi", probability=0.97, text="नमस्ते आस्ता"):
        self.calls = []
        self.language = language
        self.probability = probability
        self.text = text

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return [
            SimpleNamespace(
                text=self.text,
                avg_logprob=-0.2,
                no_speech_prob=0.02,
                compression_ratio=1.2,
            )
        ], SimpleNamespace(
            language=self.language,
            language_probability=self.probability,
        )


class FakeIndic:
    def __init__(self, text="नमस्ते आस्ता"):
        self.calls = []
        self.text = text

    def transcribe(self, audio, language=None):
        self.calls.append((audio, language))
        return self.text


def make_multilingual_engine(fake_whisper, fake_indic=None, threshold=0.70):
    engine = RecognitionEngine.__new__(RecognitionEngine)
    engine.model = fake_whisper
    engine.model_name = "medium"
    engine.beam_size = 5
    engine.language = "en"
    engine.language_detection_threshold = threshold
    engine.debug = False
    engine.last_language = None
    engine.last_language_probability = 0.0
    engine.last_backend = None
    engine.backend = "multilingual"
    engine.indic_language = "hi"
    engine.indic_decoder = "rnnt"
    engine.indic_model_id = "test"
    engine._indic = fake_indic
    engine._indic_unavailable = False
    engine._indic_error = None
    engine.language_router = STTLanguageRouter(indic_languages={"hi"})
    return engine


def test_language_router_prefers_indic_for_hindi():
    router = STTLanguageRouter(indic_languages={"hi"})

    assert router.preferred_backend("hi") == "indic"
    assert router.preferred_backend("en") == "whisper"


def test_multilingual_mode_routes_high_confidence_hindi_to_indic():
    whisper = FakeWhisper(language="hi", probability=0.97, text="नमस्ते आस्ता")
    indic = FakeIndic(text="नमस्ते आस्ता")
    engine = make_multilingual_engine(whisper, indic)

    text = engine.transcribe("audio")

    assert text == "नमस्ते आस्ता"
    assert engine.last_language == "hi"
    assert engine.last_language_probability == 0.97
    assert engine.last_backend == "indic-conformer"
    assert indic.calls == [("audio", "hi")]
    assert whisper.calls[0][1]["language"] is None


def test_multilingual_mode_keeps_whisper_when_language_detection_is_uncertain():
    whisper = FakeWhisper(language="hi", probability=0.42, text="namaste asta")
    indic = FakeIndic(text="यह नहीं चलना चाहिए")
    engine = make_multilingual_engine(whisper, indic, threshold=0.70)

    text = engine.transcribe("audio")

    assert text == "namaste asta"
    assert engine.last_backend == "whisper"
    assert indic.calls == []


def test_hindi_unicode_is_not_rejected_as_hallucination():
    engine = object.__new__(RecognitionEngine)

    assert engine._is_hallucination("नमस्ते आस्ता") is False
