def apply_presentation_patch():
    """Compatibility no-op.

    This patch used to replace ``SpeechModule.on_assistant_sentence`` so the
    full presentation text was spoken. The replacement also dropped speech
    deferral during incremental capture, code/diagnostic filtering and emoji
    removal. ``SpeechModule`` now speaks the full presentation itself, so the
    patch is no longer needed and intentionally does nothing.
    """
    return None
