"""Tests for streaming STT, Smart Turn end-of-turn and streamed Kokoro TTS."""

import time

import numpy as np
import pytest

from ai.llama_cpp_engine import LlamaCppEngine
from voice.incremental_command_engine import IncrementalCommandDetector
from voice.incremental_speech_session import IncrementalSpeechSession
from voice.recognition_engine import RecognitionEngine
from voice.smart_turn import TurnEndTracker, whisper_features
from voice.vad_engine import VADEngine

from tests.test_incremental_voice import (
    FakeApplicationManager,
    FakeMicrophone,
    FakeVADEngine,
)


# --------------------------------------------------------------------------
# Smart Turn
# --------------------------------------------------------------------------
def test_whisper_features_shape_and_left_padding():
    features = whisper_features(np.random.default_rng(0).normal(size=16000))
    assert features.shape == (80, 800)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_whisper_features_match_transformers_when_available():
    transformers = pytest.importorskip("transformers")
    audio = np.random.default_rng(1).normal(scale=0.1, size=3 * 16000).astype(np.float32)
    padded = np.pad(audio, (8 * 16000 - audio.size, 0))
    reference = transformers.WhisperFeatureExtractor(chunk_length=8)(
        padded,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    ).input_features[0]
    assert np.abs(reference - whisper_features(audio)).max() < 1e-4


class FakeDetector:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def is_complete(self, _audio):
        self.calls += 1
        return self.answers.pop(0)


def _update(tracker, started=False, ended=False, samples=512):
    return tracker.update(
        started=started, ended=ended, chunk_samples=samples, get_audio=lambda: np.zeros(10)
    )


def test_tracker_without_smart_turn_ends_on_first_vad_end():
    tracker = TurnEndTracker(None, extra_wait_samples=8000)
    assert _update(tracker) is False
    assert _update(tracker, ended=True) is True


def test_tracker_ends_immediately_when_speech_sounds_complete():
    detector = FakeDetector([True])
    tracker = TurnEndTracker(detector, extra_wait_samples=8000)
    assert _update(tracker, ended=True) is True
    assert detector.calls == 1


def test_tracker_waits_full_silence_when_speech_sounds_unfinished():
    tracker = TurnEndTracker(FakeDetector([False]), extra_wait_samples=1600)
    assert _update(tracker, ended=True) is False
    assert _update(tracker, samples=800) is False
    assert _update(tracker, samples=800) is True


def test_tracker_resumes_when_user_keeps_talking():
    detector = FakeDetector([False, True])
    tracker = TurnEndTracker(detector, extra_wait_samples=1600)
    assert _update(tracker, ended=True) is False
    assert _update(tracker, started=True) is False
    assert tracker.pending is False
    assert _update(tracker, samples=5000) is False  # still talking, no end event
    assert _update(tracker, ended=True) is True
    assert detector.calls == 2


def test_vad_new_turn_tracker_extra_wait_from_silence_settings():
    vad = object.__new__(VADEngine)
    vad.sample_rate = 16000
    vad.silence_ms = 700
    vad.iterator_silence_ms = 200
    vad.smart_turn = FakeDetector([])
    assert vad.new_turn_tracker().extra_wait_samples == 8000


def test_collect_utterance_streams_every_captured_chunk():
    class Iterator:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            self.calls += 1
            if self.calls == 1:
                return {"start": 0}
            if self.calls == 6:
                return {"end": 1}
            return None

        def reset_states(self):
            return None

    class Microphone:
        def get_chunk(self):
            time.sleep(0.08)
            return np.full(512, 0.1, dtype=np.float32)

    vad = object.__new__(VADEngine)
    vad.sample_rate = 16000
    vad.min_speech_duration = 0.10
    vad.min_rms = 0.0
    vad.min_peak = 0.0
    vad.start_chunk_rms = 0.0
    vad.pre_roll_samples = 1600
    vad.vad = Iterator()
    vad.debug = False

    streamed = []
    audio = vad.collect_utterance(Microphone(), speech_timeout=1.0, on_audio=streamed.append)
    assert audio is not None
    assert sum(len(part) for part in streamed) == audio.size


# --------------------------------------------------------------------------
# Streaming recognition
# --------------------------------------------------------------------------
def test_hindi_transcripts_are_not_rejected_as_hallucinations():
    engine = object.__new__(RecognitionEngine)
    assert engine._normalize("आज मौसम कैसा है?") == "आज मौसम कैसा है"
    assert engine._is_hallucination("आज मौसम कैसा है और क्रोम खोलो") is False
    assert engine._is_hallucination("open chrome") is False
    assert engine._is_hallucination("thank you") is True
    assert engine._is_hallucination("...") is True


class FakeStream:
    def __init__(self, partials):
        self.partials = list(partials)
        self.samples = 0
        self.cancelled = False
        self.finished = False

    def accept(self, samples):
        self.samples += len(samples)

    @property
    def partial_text(self):
        return self.partials.pop(0) if len(self.partials) > 1 else self.partials[0]

    def finish(self):
        self.finished = True
        return "open chrome and tell me the weather"

    def cancel(self):
        self.cancelled = True


class FakeStreamingRecognition:
    supports_streaming = True

    def __init__(self):
        self.stream = FakeStream(["open chrome"])
        self.transcribe_calls = 0

    def create_stream(self):
        return self.stream

    def finish_stream(self, stream):
        return stream.finish()

    def transcribe(self, *_args, **_kwargs):
        self.transcribe_calls += 1
        return ""


def test_incremental_session_uses_stream_instead_of_redecoding():
    recognition = FakeStreamingRecognition()
    session = IncrementalSpeechSession(
        vad_engine=FakeVADEngine(),
        recognition_engine=recognition,
        command_detector=IncrementalCommandDetector(
            application_manager=FakeApplicationManager(), stable_updates=2
        ),
        partial_interval_seconds=0.20,
        min_partial_seconds=0.30,
        stt_window_seconds=2.0,
    )
    committed = []
    result = session.run(
        FakeMicrophone([np.full(1600, 0.08, dtype=np.float32) for _ in range(5)]),
        should_continue=lambda: True,
        speech_timeout=1.0,
        on_commit=committed.append,
    )
    assert result is not None
    assert recognition.transcribe_calls == 0
    assert recognition.stream.finished is True
    assert recognition.stream.samples == result.audio.size
    assert committed and committed[0].text == "open Google Chrome"
    assert result.transcript == "open chrome and tell me the weather"


def test_incremental_session_cancels_stream_without_speech():
    class Iterator:
        def __call__(self, _tensor):
            return None

        def reset_states(self):
            return None

    vad = FakeVADEngine()
    vad.vad = Iterator()
    recognition = FakeStreamingRecognition()
    session = IncrementalSpeechSession(
        vad_engine=vad,
        recognition_engine=recognition,
        command_detector=IncrementalCommandDetector(application_manager=FakeApplicationManager()),
    )
    result = session.run(
        FakeMicrophone([np.zeros(1600, dtype=np.float32) for _ in range(40)]),
        should_continue=lambda: True,
        speech_timeout=0.2,
    )
    assert result is None
    assert recognition.stream.cancelled is True


def test_auto_backend_falls_back_to_whisper_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTA_NEMOTRON_MODEL_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ASTA_STT_BACKEND", "auto")

    class FakeWhisper:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("voice.recognition_engine.WhisperModel", FakeWhisper)
    engine = RecognitionEngine()
    assert engine.backend == "whisper"
    assert engine.supports_streaming is False


# --------------------------------------------------------------------------
# LLM sentence splitting
# --------------------------------------------------------------------------
def test_hindi_danda_ends_a_sentence_for_streaming_tts():
    buffer, sentence = LlamaCppEngine._emit_sentence_chunks("क्रोम खोल दिया। अब क्या")
    assert sentence == "क्रोम खोल दिया।"
    assert buffer.strip() == "अब क्या"


# --------------------------------------------------------------------------
# Kokoro streaming
# --------------------------------------------------------------------------
def test_kokoro_first_clause_split_and_hindi_detection():
    from speech.kokoro_engine import KokoroEngine

    text = "Sure, I can do that. The weather today looks sunny with a light breeze."
    assert KokoroEngine.split_first_clause(text) == [text]  # "Sure," is too short
    long_text = "I opened Chrome for you, and the weather today looks sunny with a light breeze."
    assert KokoroEngine.split_first_clause(long_text) == [
        "I opened Chrome for you,",
        "and the weather today looks sunny with a light breeze.",
    ]
    assert KokoroEngine.is_hindi("आज मौसम बहुत अच्छा है।") is True
    assert KokoroEngine.is_hindi("Chrome खोल दिया") is True
    assert KokoroEngine.is_hindi("Opened Chrome.") is False


def test_kokoro_stream_play_writes_chunks_as_they_are_generated():
    from speech.kokoro_engine import KokoroEngine

    engine = object.__new__(KokoroEngine)
    produced = [np.full(480, 0.1, dtype=np.float32) for _ in range(3)]
    engine.iter_audio = lambda text, should_continue=None, first_clause=True: iter(produced)
    written = []
    engine.write = lambda chunk, should_continue=None, on_level=None: written.append(chunk) or True
    assert engine.stream_play("hello there") is True
    assert len(written) == 3


def test_kokoro_stream_play_stops_on_interrupt():
    from speech.kokoro_engine import KokoroEngine

    engine = object.__new__(KokoroEngine)
    engine.iter_audio = lambda text, should_continue=None, first_clause=True: iter(
        [np.ones(480, dtype=np.float32)] * 4
    )
    writes = []

    def write(chunk, should_continue=None, on_level=None):
        writes.append(chunk)
        return len(writes) < 2

    engine.write = write
    assert engine.stream_play("hello") is False
    assert len(writes) == 2


# --------------------------------------------------------------------------
# English / Hindi transcript choice
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "english, hindi, expected",
    [
        # Indian-accented English that Nemotron auto-ID wrote in Devanagari.
        ("Hello Aster Open Chrome and then check the weather",
         "हेलो आस्टर ओपन क्रोम एंड देन चेक द वेदर", "en"),
        ("Play some music on Spotify and increase the volume",
         "प्ले सम म्यूजिक ऑन स्पॉटिफाई एंड इन्क्रीस द वॉल्यूम", "en"),
        ("what time is it", "व्हाट टाइम इज इट", "en"),
        ("open spotify", "ओपन स्पॉटिफाई", "en"),
        # Hindi and Hinglish.
        ("mca or kroom kholo", "आज मौसम कैसा है और क्रोम खोलो", "hi"),
        ("", "क्या हालचाल है मुझे एक जोक सुनाओ", "hi"),
        ("chrome kholo or music bajao", "क्रोम खोलो और म्यूजिक बजाओ", "hi"),
        ("volume kam karo", "वॉल्यूम कम करो", "hi"),
    ],
)
def test_choose_transcript_between_english_and_hindi(english, hindi, expected):
    from voice.language_choice import choose_transcript

    language, text = choose_transcript(english, hindi)
    assert language == expected
    assert text == (english if expected == "en" else hindi)


def test_dual_language_stream_feeds_both_and_picks_one():
    from voice.nemotron_streaming_engine import DualLanguageStream

    class Half:
        def __init__(self, text):
            self.text = text
            self.samples = 0
            self.confidence = -0.1
            self._audio = type("Q", (), {"put": lambda self, item: None})()

        def accept(self, samples):
            self.samples += len(samples)

        @property
        def partial_text(self):
            return self.text

        def finish(self, timeout=10.0):
            return self.text

    stream = object.__new__(DualLanguageStream)
    stream.english = Half("mca or kroom kholo")
    stream.hindi = Half("आज मौसम कैसा है और क्रोम खोलो")
    stream.language = None
    stream.accept(np.zeros(320, dtype=np.float32))
    assert stream.english.samples == stream.hindi.samples == 320
    assert stream.finish() == "आज मौसम कैसा है और क्रोम खोलो"
    assert stream.language == "hi"


def test_nemotron_stream_holds_engine_lock_for_every_recognizer_call():
    """sherpa-onnx corrupts results if one stream is fed while another decodes."""
    import threading

    from voice.nemotron_streaming_engine import NemotronStream

    engine = type("Engine", (), {})()
    engine.decode_lock = threading.Lock()
    violations = []

    class Stream:
        def set_option(self, *_):
            if not engine.decode_lock.locked():
                violations.append("set_option")

        def accept_waveform(self, *_):
            if not engine.decode_lock.locked():
                violations.append("accept_waveform")

        def input_finished(self):
            if not engine.decode_lock.locked():
                violations.append("input_finished")

    class Recognizer:
        def create_stream(self):
            return Stream()

        def is_ready(self, _stream):
            if not engine.decode_lock.locked():
                violations.append("is_ready")
            return False

        def get_result(self, _stream):
            return "ok"

    engine.recognizer = Recognizer()
    streams = [NemotronStream(engine, "en"), NemotronStream(engine, "hi")]
    for _ in range(20):
        for stream in streams:
            stream.accept(np.zeros(512, dtype=np.float32))
    assert [stream.finish() for stream in streams] == ["ok", "ok"]
    assert violations == []
