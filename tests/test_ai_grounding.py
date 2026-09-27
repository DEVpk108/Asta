from ai.ai_module import AIModule
from core import Kernel
from core.tools import OpenApplicationTool, ScreenshotTool


class RecordingEngine:
    def __init__(self):
        self.system_prompt = ""
        self.calls = []

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def warmup(self):
        return True

    def generate_response(self, text, on_sentence=None):
        self.calls.append(text)
        return "LLM response"


def _make_ai():
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    kernel.register_tool(ScreenshotTool(capture=lambda: {"path": "test.png"}))

    ai = AIModule(kernel)
    engine = RecordingEngine()
    ai.engine = engine
    return kernel, ai, engine


def test_unknown_name_is_never_invented():
    kernel, ai, engine = _make_ai()
    responses = []
    kernel.event_bus.subscribe("assistant_response", lambda text: responses.append(text))

    ai.initialize()
    ai.on_user_message("What's my name?")

    assert responses[-1] == "I don't know your name yet. I don't have that information stored."
    assert engine.calls == []
    ai.shutdown()


def test_capability_questions_use_live_registry():
    kernel, ai, engine = _make_ai()
    responses = []
    kernel.event_bus.subscribe("assistant_response", lambda text: responses.append(text))

    ai.initialize()
    ai.on_user_message("Can you play music?")

    assert "system.open_application" in responses[-1]
    assert "vision.screenshot" in responses[-1]
    assert engine.calls == []
    ai.shutdown()


def test_capability_prompt_contains_only_registered_tools():
    kernel, ai, engine = _make_ai()
    ai.initialize()

    assert "REGISTERED CAPABILITIES:" in engine.system_prompt
    assert "system.open_application" in engine.system_prompt
    assert "vision.screenshot" in engine.system_prompt
    assert "play_music" not in engine.system_prompt
    ai.shutdown()


def test_ai_formats_vision_inspection_success_concisely():
    from core.contracts import ToolResult

    result = ToolResult(
        success=True,
        tool="vision.inspect",
        output={
            "verified": True,
            "visual_match": True,
            "confidence": 0.95,
            "summary": "Calculator window is visible.",
        },
    )

    assert AIModule._format_tool_success(result) == (
        "I checked the screen. The requested visual condition is confirmed."
    )
