def apply_presentation_patch():
    """Speak the same presentation text that the HUD receives."""
    from speech.speech_module import SpeechModule

    if getattr(SpeechModule, "_asta_presentation_patch_applied", False):
        return

    def on_assistant_sentence(self, text):
        if not text:
            return

        is_presentation = text.startswith("Hello Sir. I’m A.S.T.A.,")
        if is_presentation:
            with self._state_lock:
                self._presentation_mode_active = True

        with self._state_lock:
            self._queued_text += 1
            was_inactive = not self._speech_active
            self._speech_active = True

        if was_inactive:
            self._interrupt_event.clear()
            self.event_bus.emit("speech_started")

        self._queue.put(text)

    SpeechModule.on_assistant_sentence = on_assistant_sentence
    SpeechModule._asta_presentation_patch_applied = True
