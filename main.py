from core.kernel import Kernel
from core.tools import OpenApplicationTool, ToolRuntimeModule

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
    tools = ToolRuntimeModule(kernel)

    # Register capabilities before the runtime starts.
    kernel.register_tool(OpenApplicationTool())

    print("===== ASTA KERNEL =====")

    kernel.register_module(ai)
    kernel.register_module(speech)
    kernel.register_module(hud)
    kernel.register_module(voice)
    kernel.register_module(tools)

    kernel.start()
    kernel.run()


if __name__ == "__main__":
    main()
