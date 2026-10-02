from collections import deque

import numpy as np

from voice.incremental_command_engine import IncrementalCommandDetector
from voice.incremental_speech_session import IncrementalSpeechSession


class FakeApplication:
    def __init__(self, name):
        self.name = name


class FakeApplicationManager:
    def __init__(self):
        self.names = {
            "chrome": "Google Chrome",
            "the notes app": "Notes",
            "arc": "Arc",
        }

    def resolve(self, target):
        key = " ".join(str(target).strip().lower().split())
        if key not in self.names:
            raise RuntimeError(f"no app: {target}")
        return FakeApplication(self.names[key])


class FakeMicrophone:
    sample_rate = 16000

    def __init__(self, chunks):
        self.chunks = deque(chunks)

    def get_chunk(self):
        if not self.chunks:
            raise RuntimeError("microphone exhausted")
        return self.chunks.popleft().copy()


class FakeVADIterator:
    def __init__(self):
        self.calls = 0

    def reset_states(self):
        self.calls = 0

    def __call__(self, _tensor):
        self.calls += 1
        if self.calls == 1:
            return {"start": 0}
        if self.calls == 5:
            return {"end": 1}
        return None


class FakeVADEngine:
    sample_rate = 16000
    min_speech_duration = 0.30
    min_rms = 0.0
    min_peak = 0.0
    start_chunk_rms = 0.0
    pre_roll_samples = 12000

    def __init__(self):
        self.vad = FakeVADIterator()

    @staticmethod
    def is_speech_started(event):
        return event is not None and "start" in event

    @staticmethod
    def is_speech_ended(event):
        return event is not None and "end" in event


class FakeRecognition:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return "open chrome"


def test_detector_requires_two_stable_partials_before_commit():
    detector = IncrementalCommandDetector(
        application_manager=FakeApplicationManager(),
        stable_updates=2,
    )

    assert detector.observe("open") is None

    assert detector.observe("open chrome") is None

    commit = detector.observe("open chrome")
    assert commit is not None
    assert commit.text == "open Google Chrome"
    assert commit.entities == {
        "action": "open",
        "target": "Google Chrome",
    }


def test_detector_early_commit_accepts_natural_open_up_phrase():
    detector = IncrementalCommandDetector(
        application_manager=FakeApplicationManager(),
        stable_updates=1,
    )

    commit = detector.observe("okay can you open up the notes app for me")
    assert commit is not None
    assert commit.text == "open Notes"
    assert commit.source_text == "open up the notes app"


def test_detector_does_not_turn_follow_up_words_into_an_app_name():
    detector = IncrementalCommandDetector(
        application_manager=FakeApplicationManager(),
        stable_updates=1,
    )

    assert detector.observe("open chrome and") is not None
    # The committed command above is the safe prefix; a second observation
    # containing only discourse must not create another action.
    assert detector.observe("open chrome and once youre there") is None


def test_finalize_returns_only_uncommitted_follow_up():
    detector = IncrementalCommandDetector(
        application_manager=FakeApplicationManager(),
        stable_updates=1,
    )

    commit = detector.observe("open chrome")
    assert commit is not None

    remainder = detector.finalize(
        "okay can you open chrome and once you're there search for Christopher Nolan"
    )
    assert remainder == "search for Christopher Nolan"


def test_incremental_session_commits_before_final_end_of_speech():
    microphone = FakeMicrophone(
        [np.full(1600, 0.08, dtype=np.float32) for _ in range(5)]
    )
    recognition = FakeRecognition()
    vad = FakeVADEngine()
    detector = IncrementalCommandDetector(
        application_manager=FakeApplicationManager(),
        stable_updates=2,
    )
    session = IncrementalSpeechSession(
        vad_engine=vad,
        recognition_engine=recognition,
        command_detector=detector,
        partial_interval_seconds=0.20,
        min_partial_seconds=0.30,
        stt_window_seconds=2.0,
    )

    committed = []
    result = session.run(
        microphone,
        should_continue=lambda: True,
        speech_timeout=1.0,
        on_commit=committed.append,
    )

    assert result is not None
    assert committed
    assert committed[0].text == "open Google Chrome"
    assert result.commits == tuple(committed)
    assert result.remainder == ""
    assert len(recognition.calls) >= 3
