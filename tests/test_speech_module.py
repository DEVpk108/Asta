import queue

from speech import speech_module


def make_speech_module():
    module = object.__new__(speech_module.SpeechModule)
    module._queue = queue.Queue()
    module.coalesce_window = 0.01
    return module


def test_coalesces_rapid_sentence_chunks():
    module = make_speech_module()
    module._queue.put("Of course!")
    module._queue.put("I’m here to help.")

    first = module._queue.get()
    combined = module._get_coalesced_text(first)
    module._queue.task_done()

    assert combined == "Of course! I’m here to help."
    assert module._queue.empty()


def test_coalescer_marks_consumed_items_done():
    module = make_speech_module()
    module._queue.put("First.")
    module._queue.put("Second.")

    first = module._queue.get()
    module._get_coalesced_text(first)
    module._queue.task_done()

    # queue.join() returns immediately only when every consumed item was
    # accounted for with task_done().
    module._queue.join()
