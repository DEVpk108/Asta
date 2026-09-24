from voice.voice_module import VoiceModule


def test_barge_in_stop_phrase_is_detected():
    interrupted, tail = VoiceModule._extract_interrupt_tail("Okay, okay, stop.")
    assert interrupted is True
    assert tail == ""


def test_barge_in_supports_direct_stop():
    interrupted, tail = VoiceModule._extract_interrupt_tail("stop")
    assert interrupted is True
    assert tail == ""


def test_barge_in_supports_supported_stop_command():
    interrupted, tail = VoiceModule._extract_interrupt_tail("stop speaking")
    assert interrupted is True
    assert tail == ""


def test_barge_in_does_not_trigger_on_unrelated_stop_sentence():
    interrupted, tail = VoiceModule._extract_interrupt_tail(
        "Please stop the process."
    )
    assert interrupted is False
    assert tail == ""


def test_barge_in_does_not_trigger_on_normal_command():
    interrupted, tail = VoiceModule._extract_interrupt_tail("open camera")
    assert interrupted is False
    assert tail == ""



def test_post_tts_preroll_is_preserved_for_immediate_user_speech():
    import numpy as np

    class Microphone:
        def get_buffer(self):
            return np.array([0.1, -0.2, 0.3], dtype=np.float32)

    voice = object.__new__(VoiceModule)
    voice.microphone = Microphone()
    voice._post_tts_seed_pending = True

    seed = voice._take_post_tts_seed()

    assert seed is not None
    assert np.array_equal(seed, np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert voice._post_tts_seed_pending is False


def test_post_tts_preroll_is_consumed_only_once():
    class Microphone:
        def get_buffer(self):
            raise AssertionError("microphone should not be sampled twice")

    voice = object.__new__(VoiceModule)
    voice.microphone = Microphone()
    voice._post_tts_seed_pending = False

    assert voice._take_post_tts_seed() is None



def test_speech_finished_uses_short_post_tts_settle_window():
    import threading
    import time

    class Microphone:
        def __init__(self):
            self.cleared = False

        def clear_buffer(self):
            self.cleared = True

    voice = object.__new__(VoiceModule)
    voice.microphone = Microphone()
    voice._tts_active = True
    voice._tts_guard_until = 0.0
    voice._post_tts_seed_pending = False
    voice._post_tts_guard_seconds = 0.15
    voice._conversation_active = True
    voice._manual_conversation = False
    voice._last_interaction = 0.0
    voice._barge_stop = threading.Event()
    voice._barge_thread = None
    voice._stop_barge_listener = lambda: None

    before = time.monotonic()
    voice._on_speech_finished()

    assert voice._tts_active is False
    assert voice._post_tts_seed_pending is True
    assert 0.05 <= voice._tts_guard_until - before <= 0.30
    assert voice.microphone.cleared is True
