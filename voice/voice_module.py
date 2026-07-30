from core.module import Module
from .recognition_engine import RecognitionEngine
from .microphone_engine import MicrophoneEngine
from .vad_engine import VADEngine
import threading

class VoiceModule(Module):
    def __init__(self, kernel):
        super().__init__(
            name = "VoiceModule",
            event_bus= kernel.event_bus,
            kernel=kernel
        )
        self.microphone = MicrophoneEngine()
        self.vad = VADEngine()
        self.recognition = RecognitionEngine()
        
    def initialize(self):

        self.microphone.start()

        threading.Thread(
            target=self.listen_loop,
            daemon=True
        ).start()
        

    def listen_loop(self):
        while True:

            audio = self.vad.collect_utterance(self.microphone)

            if audio is None:
                continue

            text = self.recognition.transcribe(audio)

            self.event_bus.emit(
               "user_message",
            text=text
            )
            
    def shutdown(self):

        self.microphone.stop()
