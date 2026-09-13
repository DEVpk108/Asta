from core.kernel import Kernel
from core.tools import (
    AudioControlTool,
    CloseApplicationTool,
    LaunchApplicationTool,
    OpenApplicationTool,
    OpenScreenshotTool,
    RunCommandTool,
    ScreenshotTool,
    StartProcessTool,
    StopProcessTool,
    ToolRuntimeModule,
)

from speech.speech_module import SpeechModule
from speech.presentation_patch import apply_presentation_patch
from hud.hud_module import HUDModule
from ai.ai_module import AIModule
from ai.runtime_patch import apply_ai_runtime_patch
from input.text_input_module import TextInputModule
from voice.voice_module import VoiceModule
from vision.screenshot_backend import capture_screenshot


def main():
    # Apply small runtime compatibility patches before module instances are created.
    apply_ai_runtime_patch()
    apply_presentation_patch()

    kernel = Kernel()

    speech = SpeechModule(kernel)
    hud = HUDModule(kernel)
    ai = AIModule(kernel)
    text_input = TextInputModule(kernel)
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
        OpenScreenshotTool(),
        AudioControlTool("mute"),
        AudioControlTool("unmute"),
    ):
        kernel.register_tool(tool)

    print("===== ASTA KERNEL =====")

    kernel.register_module(ai)
    kernel.register_module(speech)
    kernel.register_module(hud)
    kernel.register_module(text_input)
    kernel.register_module(voice)
    kernel.register_module(tools)

    kernel.start()
    kernel.run()


if __name__ == "__main__":
    main()
