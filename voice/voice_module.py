from core.module import Module

from .recognition_engine import RecognitionEngine
from .microphone_engine import MicrophoneEngine
from .vad_engine import VADEngine
from .wakeword_engine import WakeWordEngine

import threading


class VoiceModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="VoiceModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.microphone = MicrophoneEngine()
        self.wakeword = WakeWordEngine(
            model_path="generated/models/asta.onnx",
            threshold=0.5,
            debug=True,
        )
        self.vad = VADEngine()
        self.recognition = RecognitionEngine()

        self.running = False

    def initialize(self):

        print("[Voice] Initializing...")

        self.microphone.start()

        self.running = True

        threading.Thread(
            target=self.listen_loop,
            daemon=True,
            name="VoiceListenLoop",
        ).start()

        print("[Voice] Ready.")

    def listen_loop(self):

        while self.running:

            # -------------------------------------------------
            # 1. Wait for one of the ASTA wake phrases
            # -------------------------------------------------

            wakeword_audio = self.wakeword.wait_for_wakeword(
                self.microphone
            )

            if not self.running:
                break

            print("[Voice] ASTA activated.")

            # -------------------------------------------------
            # 2. Wait for the actual user command
            # -------------------------------------------------

            audio = self.vad.collect_utterance(
                self.microphone,
                initial_audio=wakeword_audio,
                speech_timeout=3.0,
            )

            if audio is None:
                print("[Voice] No command detected.")
                continue

            # -------------------------------------------------
            # 3. Speech -> text
            # -------------------------------------------------

            text = self.recognition.transcribe(audio)

            if not text:
                continue

            print(f"[Voice] User: {text}")

            # -------------------------------------------------
            # 4. Send command to ASTA
            # -------------------------------------------------

            self.event_bus.emit(
                "user_message",
                text=text,
            )

    def shutdown(self):

        self.running = False

        try:
            self.microphone.stop()
        except Exception:
            pass

        print("[Voice] Shutdown.")