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
    from ai.ai_module import AIModule

    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    kernel.register_tool(ScreenshotTool(capture=lambda: {"path": "test.png"}))

    ai = AIModule(kernel)
    engine = RecordingEngine()
    ai.engine = engine
    return kernel, ai, engine


def test_present_yourself_uses_dedicated_presentation():
    kernel, ai, engine = _make_ai()
    responses = []
    kernel.event_bus.subscribe("assistant_response", lambda text: responses.append(text))

    ai.initialize()
    ai.on_user_message("Okay, present yourself.")

    assert engine.calls == []
    assert responses
    assert any("Mr. PRASANT KUMAR" in response for response in responses)
    assert any("local-first AI engineering assistant" in response for response in responses)
    ai.shutdown()
