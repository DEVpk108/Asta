from core import Kernel
from core.tools import ScreenshotTool, ToolRuntimeModule
from ai.runtime_patch import (
    _extract_post_response_screenshot,
    _is_creator_identity_question,
    apply_ai_runtime_patch,
)
from speech.presentation_patch import apply_presentation_patch


class RecordingEngine:
    def __init__(self):
        self.calls = []
        self.system_prompt = ""

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def warmup(self):
        return True

    def generate_response(self, text, on_sentence=None):
        self.calls.append(text)
        if on_sentence:
            on_sentence("Here is a joke.")
        return "Here is a joke."


def test_post_response_screenshot_parser_keeps_only_the_response_prompt():
    assert _extract_post_response_screenshot(
        "Okay tell me a joke then take a screenshot"
    ) == "Okay tell me a joke"
    assert _extract_post_response_screenshot(
        "Tell me a fact, and then capture the screen shot."
    ) == "Tell me a fact"
    assert _extract_post_response_screenshot("Open Chrome then take screenshot") is not None


def test_creator_identity_patch_covers_natural_variants():
    assert _is_creator_identity_question("Who is your creator?")
    assert _is_creator_identity_question("Who is behind ASTA?")
    assert _is_creator_identity_question("Who developed you?")
    assert not _is_creator_identity_question("Who are you?")


def test_mixed_request_generates_response_then_executes_screenshot():
    from ai.ai_module import AIModule

    apply_ai_runtime_patch()

    kernel = Kernel()
    captured = []
    kernel.register_tool(ScreenshotTool(capture=lambda: captured.append("shot.png") or {"path": "shot.png"}))

    ai = AIModule(kernel)
    engine = RecordingEngine()
    ai.engine = engine

    tools = ToolRuntimeModule(kernel)
    kernel.register_module(ai)
    kernel.register_module(tools)
    kernel.start()
    try:
        ai.on_user_message("Okay tell me a joke then take a screenshot")

        assert engine.calls == ["Okay tell me a joke"]
        assert captured == ["shot.png"]
    finally:
        kernel.shutdown()


def test_presentation_patch_does_not_replace_sentences_with_a_short_script():
    from speech.speech_module import SpeechModule

    apply_presentation_patch()
    assert getattr(SpeechModule, "_asta_presentation_patch_applied", False) is True
