from core.kernel import Kernel

from speech.speech_module import SpeechModule
from hud.hud_module import HUDModule
from ai.ai_module import AIModule


def main():
    kernel = Kernel()

    speech = SpeechModule(kernel)
    hud = HUDModule(kernel)
    ai = AIModule(kernel)
    
    print("===== ASTA KERNEL =====")

    kernel.register_module(ai)
    
    kernel.register_module(speech)
    
    kernel.register_module(hud)
    

    kernel.start()
    
    kernel.event_bus.emit(
        "user_message",
        text="Hello ASTA"
    )


if __name__ == "__main__":
    main()