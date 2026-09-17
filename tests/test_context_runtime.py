from core import Kernel
from core.tools import OpenApplicationTool
from ai.ai_module import AIModule
from ai.context_runtime_patch import apply_context_runtime_patch


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def generate_response(self, text, on_sentence=None):
        self.calls.append(text)
        return "context-aware response"


def test_context_runtime_patch_enriches_model_input():
    kernel = Kernel()
    kernel.register_tool(OpenApplicationTool())
    kernel.memory_context = "The user is building A.S.T.A. locally."
    kernel.create_task(
        "debug calculator launch",
        pending_steps=["open calculator"],
    )

    engine = RecordingEngine()
    ai = AIModule(kernel)
    ai.engine = engine

    original_generate_response = AIModule._generate_response
    original_patch_flag = getattr(
        AIModule,
        "_asta_context_runtime_patch_applied",
        None,
    )

    try:
        apply_context_runtime_patch()
        ai._generate_response("open calculator")

        assert len(engine.calls) == 1
        prompt = engine.calls[0]
        assert "A.S.T.A. RUNTIME CONTEXT" in prompt
        assert "debug calculator launch" in prompt
        assert "The user is building A.S.T.A. locally." in prompt
        assert "system.open_application" in prompt
        assert "USER REQUEST:\nopen calculator" in prompt
    finally:
        AIModule._generate_response = original_generate_response
        if original_patch_flag is None:
            try:
                delattr(AIModule, "_asta_context_runtime_patch_applied")
            except AttributeError:
                pass
        else:
            AIModule._asta_context_runtime_patch_applied = original_patch_flag
