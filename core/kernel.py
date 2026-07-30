from .event_bus import EventBus

class Kernel:
    def __init__(self):
        self.event_bus = EventBus()
        self.modules = []
        
    def register_module(self, module):
        self.modules.append(module)
        
    def start(self):
        for module in self.modules:
            module.initialize()
        print("Starting Kernel...")
    
    def shutdown(self):
        for module in self.modules:
            module.shutdown()
    
    def register_module(self, module):
       self.modules.append(module)
       print(f"[Kernel] Registered {module.name}")
                       
