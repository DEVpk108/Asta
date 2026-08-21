# voice/voice_module.py

import threading

from core.module import Module

from .microphone_engine import MicrophoneEngine
from .wakeword_engine import WakeWordEngine
from .vad_engine import VADEngine
from .recognition_engine_2 import RecognitionEngine


class VoiceModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="Voice",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.microphone = MicrophoneEngine()
        self.wakeword = WakeWordEngine()
        self.vad = VADEngine()
        self.recognition = RecognitionEngine()

        self._running = False
        self._thread = None

    def initialize(self):
        print("[Voice] Initializing...")

        self.microphone.start()

        self._running = True

        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
        )

        self._thread.start()

        print("[Voice] Ready")

    def shutdown(self):
        print("[Voice] Shutting down...")

        self._running = False

        if self.microphone:
            self.microphone.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        print("[Voice] Stopped")

    def _listen_loop(self):

        while self._running:

            try:
                # -------------------------------------------------
                # Wait for one of the three wake words.
                # -------------------------------------------------

                initial_audio = (
                    self.wakeword.wait_for_wakeword(
                        self.microphone
                    )
                )

                if not self._running:
                    break

                # -------------------------------------------------
                # Capture the user's command.
                # -------------------------------------------------

                audio = self.vad.collect_utterance(
                    self.microphone,
                    initial_audio,
                    speech_timeout=3,
                )

                if audio is None:
                    continue

                if not self._running:
                    break

                # -------------------------------------------------
                # Speech → text.
                # -------------------------------------------------

                text = self.recognition.transcribe(audio)

                if not text:
                    continue

                print(f"[Voice] User: {text}")

                # -------------------------------------------------
                # Send the recognized command into ASTA.
                # -------------------------------------------------

                self.event_bus.emit(
                    "user_message",
                    text=text,
                )

            except Exception as exc:

                print(
                    f"[Voice] Error: "
                    f"{type(exc).__name__}: {exc}"
                )