from core.module import Module
from .openai_engine import AIEngine

class AIModule(Module):
    def __init__(self,kernel):
        super().__init__(
            name="AIModule",
                event_bus=kernel.event_bus,
                kernel=kernel
        )
        self.engine = AIEngine()
    def initialize(self):
        self.event_bus.subscribe(
            "user_message",
            self.on_user_message
        )
        
    def on_user_message(self,text):
        response = self.engine.generate_response(text)
        print(f"User:{text}")
        print(f"AI: Echo:{text}")
        
        self.event_bus.emit(
            "assistant_response",
            text=response
        )
        