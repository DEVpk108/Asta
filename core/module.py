class Module:
    def __init__(self,name,event_bus,kernel):
        self.name = name
        self.event_bus = event_bus
        self.kernel= kernel
    
    
    def initialize(self):
        pass
    def shutdown(self):
        pass
        