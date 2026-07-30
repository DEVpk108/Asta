from core import Modules

class DummySpeechModule(Modules):
    def __init__(self, kernel):
        super().__init__(
            
        name="Speech",
            event_bus=kernel.event_bus,
            kernel=kernel
        )
        
    def initialize(self):
        print("[Speech] Initialized")

    def shutdown(self):
        print("[Speech] Shutdown")


