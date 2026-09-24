from types import SimpleNamespace

from voice.recognition_engine import RecognitionEngine


class FakeWhisper:
    def __init__(self, segments=None):
        self.calls = []
        self.segments = segments or [
            SimpleNamespace(
                text="namaste ASTA",
                avg_logprob=-0.2,
                no_speech_prob=0.02,
                compression_ratio=1.2,
            )
        ]

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.segments, SimpleNamespace(
            language="hi",
            language_probability=0.97,
        )


def make_engine(fake_model, language=None):
    engine = RecognitionEngine.__new__(RecognitionEngine)
    engine.model = fake_model
    engine.model_name = "medium"
    engine.beam_size = 5
    engine.language = language
    engine.debug = False
    engine.last_language = None
    engine.last_language_probability = 0.0
    engine.backend = "whisper"
    return engine


def test_recognition_does_not_force_english_or_seed_prompt_text():
    fake_model = FakeWhisper()
    engine = make_engine(fake_model, language=None)

    text = engine.transcribe("audio")

    assert text == "namaste ASTA"
    kwargs = fake_model.calls[0][1]
    assert kwargs["language"] is None
    assert "initial_prompt" not in kwargs
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["vad_filter"] is False


def test_unreliable_segments_are_filtered_without_dropping_good_speech():
    fake_model = FakeWhisper(
        segments=[
            SimpleNamespace(
                text="The speaker is using English.",
                avg_logprob=-1.4,
                no_speech_prob=0.91,
                compression_ratio=1.1,
            ),
            SimpleNamespace(
                text="open camera",
                avg_logprob=-0.25,
                no_speech_prob=0.03,
                compression_ratio=1.2,
            ),
        ]
    )
    engine = make_engine(fake_model, language="en")

    text = engine.transcribe("audio")

    assert text == "open camera"


def test_obvious_whisper_hallucination_phrase_is_rejected():
    engine = object.__new__(RecognitionEngine)

    assert engine._is_hallucination("The speaker is using English.") is True
    assert engine._is_hallucination("open camera") is False


def test_strict_recognition_rejects_high_no_speech_segments():
    engine = object.__new__(RecognitionEngine)

    noisy = SimpleNamespace(
        text="Thank you.",
        avg_logprob=-0.92,
        no_speech_prob=0.83,
        compression_ratio=0.56,
    )

    assert engine._segment_is_unreliable(noisy, strict=True) is True
    assert engine._segment_is_unreliable(noisy, strict=False) is False



def test_hallucination_filter_rejects_punctuated_acknowledgement():
    engine = object.__new__(RecognitionEngine)

    assert engine._is_hallucination("Okay. Thank you.") is True


def test_low_confidence_whisper_decode_can_retry_and_select_better_result():
    first = [
        SimpleNamespace(
            text="Only Hanuman Chalisa on Spotify.",
            avg_logprob=-0.75,
            no_speech_prob=0.67,
            compression_ratio=0.80,
        )
    ]
    retry = [
        SimpleNamespace(
            text="Play Hanuman Chalisa on Spotify.",
            avg_logprob=-0.30,
            no_speech_prob=0.08,
            compression_ratio=0.75,
        )
    ]

    class RetryWhisper:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, **kwargs):
            self.calls += 1
            segments = first if self.calls == 1 else retry
            return segments, SimpleNamespace(
                language="en",
                language_probability=0.99,
            )

    engine = make_engine(RetryWhisper(), language="en")

    assert engine.transcribe("audio") == "Play Hanuman Chalisa on Spotify."
    assert len(engine.model.calls) == 2
