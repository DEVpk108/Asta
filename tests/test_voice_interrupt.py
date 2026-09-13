from voice.interrupt_patch import _extract_interrupt_tail


def test_barge_in_stop_phrase_is_detected():
    interrupted, tail = _extract_interrupt_tail("Okay, okay, stop.")
    assert interrupted is True
    assert tail == ""


def test_barge_in_supports_whisper_inserted_filler():
    interrupted, tail = _extract_interrupt_tail("Okay, please stop.")
    assert interrupted is True
    assert tail == ""


def test_barge_in_preserves_command_after_stop():
    interrupted, tail = _extract_interrupt_tail(
        "Okay, okay, stop, open camera."
    )
    assert interrupted is True
    assert tail == "open camera"


def test_barge_in_preserves_command_after_fuzzy_stop():
    interrupted, tail = _extract_interrupt_tail(
        "Okay, please stop, open camera."
    )
    assert interrupted is True
    assert tail == "open camera"


def test_barge_in_supports_direct_stop():
    interrupted, tail = _extract_interrupt_tail("stop")
    assert interrupted is True
    assert tail == ""


def test_barge_in_does_not_trigger_on_unrelated_stop_sentence():
    interrupted, tail = _extract_interrupt_tail(
        "Please stop the process."
    )
    assert interrupted is False
    assert tail == ""


def test_barge_in_does_not_trigger_on_normal_command():
    interrupted, tail = _extract_interrupt_tail("open camera")
    assert interrupted is False
    assert tail == ""
