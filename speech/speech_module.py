from core.module import Module
from .polly_engine import PollyEngine

class SpeechModule(Module):
    def __init__(self, kernel):
        super().__init__(
          
        name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel
        )
        self.engine = PollyEngine()
    
    def initialize(self):
        self.event_bus.subscribe(
        "assistant_response",
        self.on_assistant_response
    )
    
    def on_assistant_response(self, text):
        print(f"[Speech] {text}")
        self.engine.speak(text)
        
        