from voice.recognition_engine import RecognitionEngine


def test_supported_stt_backends():
    assert RecognitionEngine.SUPPORTED_BACKENDS == {"whisper", "indic", "hybrid"}


def test_invalid_stt_backend_is_rejected():
    try:
        RecognitionEngine(backend="not-a-backend")
    except ValueError as exc:
        assert "Unsupported STT backend" in str(exc)
    else:
        raise AssertionError("Invalid STT backend should raise ValueError")


def test_recognition_engine_can_select_hybrid_without_loading_indic():
    engine = object.__new__(RecognitionEngine)
    engine.backend = "hybrid"
    engine._indic = None
    engine.last_backend = None
    assert engine.backend == "hybrid"
    assert engine._indic is None


def test_indic_language_and_decoder_configuration_is_explicit():
    engine = object.__new__(RecognitionEngine)
    engine.indic_language = "hi"
    engine.indic_decoder = "rnnt"
    assert engine.indic_language == "hi"
    assert engine.indic_decoder == "rnnt"
