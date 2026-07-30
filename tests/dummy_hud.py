from core import Modules

class DummyHUDModule(Modules):
     def __init__(self, kernel):
        super().__init__(
            
        name="HUD",
            event_bus=kernel.event_bus,
            kernel=kernel
        )
        
     def initialize(self):
        print("[HUD] Initialized")

     def shutdown(self):
        print("[HUD] Shutdown")