from core.kernel import Kernel

from speech.speech_module import SpeechModule
from hud.hud_module import HUDModule
from ai.ai_module import AIModule
from voice.voice_module import VoiceModule


def main():
    kernel = Kernel()

    speech = SpeechModule(kernel)
    hud = HUDModule(kernel)
    ai = AIModule(kernel)
    voice = VoiceModule(kernel)
    
    print("===== ASTA KERNEL =====")
    
    

    kernel.register_module(ai)
    
    kernel.register_module(speech)
    
    kernel.register_module(hud)
    
    kernel.register_module(voice)
    
    

    kernel.start()
    kernel.run()
    
    


if __name__ == "__main__":
    main()