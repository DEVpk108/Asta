from core import Kernel
from .dummy_speech import DummySpeechModule
from .dummy_hud import DummyHUDModule


print("===== ASTA KERNEL TEST =====")
print("Creating Kernel...")
kernel = Kernel()
print("Kernel Ready")


speech = DummySpeechModule(kernel)
hud = DummyHUDModule(kernel)

kernel.register_module(speech)
print("Registering Speech Module...")
kernel.register_module(hud)
print("Registering HUD Module...")

print("Starting Kernel...")
kernel.start()
print("Shutting down Kernel...")
kernel.shutdown()

print("kernel Stopped.")