from core.kernel import Kernel
from core.tools import (
    AudioControlTool,
    CloseApplicationTool,
    LaunchApplicationTool,
    OpenApplicationTool,
    RunCommandTool,
    ScreenshotTool,
    StartProcessTool,
    StopProcessTool,
    ToolRuntimeModule,
)

from speech.speech_module import SpeechModule
from hud.hud_module import HUDModule
from ai.ai_module import AIModule
from voice.voice_module import VoiceModule
from vision.screenshot_backend import capture_screenshot


def main():
    kernel = Kernel()

    speech = SpeechModule(kernel)
    hud = HUDModule(kernel)
    ai = AIModule(kernel)
    voice = VoiceModule(kernel)
    tools = ToolRuntimeModule(kernel)

    # Register capabilities before the runtime starts.
    for tool in (
        OpenApplicationTool(),
        LaunchApplicationTool(),
        StartProcessTool(),
        RunCommandTool(),
        StopProcessTool(),
        CloseApplicationTool(),
        ScreenshotTool(capture=capture_screenshot),
        AudioControlTool("mute"),
        AudioControlTool("unmute"),
    ):
        kernel.register_tool(tool)

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
