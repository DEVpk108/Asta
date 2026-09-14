import os
import shutil
import subprocess
import sys
from pathlib import Path

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
from ai.final_runtime_patch import apply_final_runtime_patch
from input.text_input_module import TextInputModule
from voice.voice_module import VoiceModule
from vision.screenshot_backend import capture_screenshot


_HUD_PROCESS = None


def _text_input_enabled():
    """Keep the legacy terminal text adapter opt-in for the voice-first runtime."""
    return os.getenv("ASTA_ENABLE_TEXT_INPUT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _start_hud():
    """Launch the Electron HUD as part of the A.S.T.A. runtime."""
    global _HUD_PROCESS

    if os.getenv("ASTA_DISABLE_HUD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print("[HUD] Auto-launch disabled (ASTA_DISABLE_HUD=1).", flush=True)
        return None

    hud_dir = Path(__file__).resolve().parent / "hud"
    package_json = hud_dir / "package.json"
    if not package_json.exists():
        print(f"[HUD] package.json not found: {package_json}", flush=True)
        return None

    # Windows exposes npm as a .cmd shim. Passing npm.cmd directly to
    # CreateProcess avoids WinError 193 that occurs when the wrong shim is
    # selected by shutil.which() from a Python subprocess.
    npm = None
    if os.name == "nt":
        npm = shutil.which("npm.cmd")
    if npm is None:
        npm = shutil.which("npm")
    if npm is None:
        print("[HUD] npm was not found; HUD auto-launch skipped.", flush=True)
        return None

    try:
        _HUD_PROCESS = subprocess.Popen(
            [npm, "start"],
            cwd=str(hud_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        print(
            f"[HUD] Electron HUD launched (PID {_HUD_PROCESS.pid}).",
            flush=True,
        )
        return _HUD_PROCESS
    except OSError as exc:
        _HUD_PROCESS = None
        print(
            f"[HUD] Failed to launch Electron HUD: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _stop_hud():
    """Stop the HUD child process when A.S.T.A. shuts down."""
    global _HUD_PROCESS

    process = _HUD_PROCESS
    _HUD_PROCESS = None
    if process is None or process.poll() is not None:
        return

    print("[HUD] Shutting down Electron HUD...", flush=True)
    try:
        process.terminate()
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def main():
    # Apply small runtime compatibility patches before module instances are created.
    apply_ai_runtime_patch()
    apply_final_runtime_patch()
    apply_presentation_patch()

    # The HUD is part of the A.S.T.A. application now. Launch it first so the
    # Electron window can connect while the Kernel initializes. The HUD itself
    # remains hidden until it receives the Kernel's READY lifecycle signal.
    _start_hud()

    kernel = Kernel()

    hud = HUDModule(kernel)
    speech = SpeechModule(kernel)
    ai = AIModule(kernel)
    voice = VoiceModule(kernel)
    tools = ToolRuntimeModule(kernel)
    text_input = TextInputModule(kernel) if _text_input_enabled() else None

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

    # HUD subscribes before AIModule so synchronous user_message dispatch
    # reaches THINKING before model inference begins.
    kernel.register_module(hud)
    kernel.register_module(ai)
    kernel.register_module(speech)
    kernel.register_module(voice)
    kernel.register_module(tools)

    if text_input is not None:
        kernel.register_module(text_input)
        print(
            "[Text] Legacy terminal input enabled "
            "(ASTA_ENABLE_TEXT_INPUT=1).",
            flush=True,
        )
    else:
        print(
            "[Text] Legacy terminal input disabled; voice-first runtime active.",
            flush=True,
        )

    try:
        kernel.start()

        if text_input is not None:
            text_input.start_input()

        kernel.run()
    finally:
        _stop_hud()


if __name__ == "__main__":
    main()
