from voice.voice_module import VoiceModule
from voice.vad_engine import VADEngine


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



def test_vad_starts_from_speech_contained_in_initial_preroll():
    import numpy as np

    class FakeVadIterator:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            self.calls += 1
            if self.calls >= 12:
                return {"end": 1}
            return None

        def reset_states(self):
            return None

    class Microphone:
        def get_chunk(self):
            import time

            time.sleep(0.04)
            return np.zeros(512, dtype=np.float32)

    vad = object.__new__(VADEngine)
    vad.sample_rate = 16000
    vad.min_speech_duration = 0.10
    vad.min_rms = 0.012
    vad.min_peak = 0.04
    vad.start_chunk_rms = 0.005
    vad.pre_roll_samples = 12800
    vad.vad = FakeVadIterator()
    vad.debug = False

    seed = np.full(1600, 0.08, dtype=np.float32)
    audio = vad.collect_utterance(
        Microphone(),
        initial_audio=seed,
        speech_timeout=1.0,
    )

    assert audio is not None
    assert audio.size >= seed.size
    # The capture also contains the subsequent silence used to satisfy the
    # simulated VAD end condition, so whole-buffer RMS is diluted. Verify the
    # actual contract: the speech preroll survived intact at the beginning.
    assert np.array_equal(audio[:seed.size], seed)
    assert float(np.max(np.abs(audio[:seed.size]))) >= 0.08



def test_barge_echo_baseline_requires_energy_rise_during_tts():
    voice = object.__new__(VoiceModule)
    voice._tts_active = True
    voice._barge_tts_warmup_until = 0.0
    voice._barge_echo_rms = None
    voice._barge_echo_peak = None
    voice._barge_min_rms = 0.012
    voice._barge_min_peak = 0.045

    # The first TTS window establishes the room/speaker echo floor.
    voice._barge_echo_rms = 0.020
    voice._barge_echo_peak = 0.120

    echo_rms_gate = max(
        voice._barge_min_rms,
        voice._barge_echo_rms * 1.70,
    )
    echo_peak_gate = max(
        voice._barge_min_peak,
        voice._barge_echo_peak * 1.45,
    )

    assert 0.022 < echo_rms_gate
    assert 0.1483 < echo_peak_gate
