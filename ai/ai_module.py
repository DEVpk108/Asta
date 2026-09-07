from core.module import Module
from core.contracts import IntentType, IntentResult, ToolResult
from core.tools import ToolRequestBuilder

from .openai_engine import AIEngine


class AIModule(Module):

    def __init__(self, kernel):
        super().__init__(
            name="AIModule",
            event_bus=kernel.event_bus,
            kernel=kernel,
        )

        self.engine = AIEngine()
        self.tool_request_builder = ToolRequestBuilder(kernel.tool_registry)

    def initialize(self):
        print("[AI] Initializing...", flush=True)
        self.event_bus.subscribe("user_message", self.on_user_message)
        self.event_bus.subscribe("tool_result", self.on_tool_result)
        self.event_bus.subscribe("tool_confirmation_required", self.on_tool_confirmation_required)
        print("[AI] Ready", flush=True)

    def shutdown(self):
        self.event_bus.unsubscribe("user_message", self.on_user_message)
        self.event_bus.unsubscribe("tool_result", self.on_tool_result)
        self.event_bus.unsubscribe("tool_confirmation_required", self.on_tool_confirmation_required)
        print("[AI] Stopped", flush=True)

    def on_user_message(self, text):
        if not text:
            return
        print(f"[AI] User: {text}", flush=True)
        result: IntentResult = self.kernel.intent_router.analyze(text)
        print(f"[AI] Intent: {result.intent.value} (confidence={result.confidence:.2f}, classifier={result.classifier})", flush=True)
        if result.intent == IntentType.COMMAND:
            self._handle_command_intent(result)
            return
        if result.intent == IntentType.MEMORY:
            self.event_bus.emit("memory_request", intent=result)
            return
        self._generate_response(text)

    def _handle_command_intent(self, intent: IntentResult):
        try:
            request = self.tool_request_builder.build(intent)
        except ValueError as exc:
            print(f"[AI] Unable to build tool request: {exc}", flush=True)
            self.event_bus.emit("assistant_response", text=f"I couldn't map that command to an available tool: {exc}")
            return
        print(f"[AI] Selected tool: {request.tool} (request_id={request.request_id})", flush=True)
        self.event_bus.emit("tool_request", request=request)

    def on_tool_confirmation_required(self, request, reason):
        action = request.metadata.get("action")
        target = request.arguments.get("target")
        if action and target:
            text = f"I need your confirmation before I {action} {target}."
        elif target:
            text = f"I need your confirmation before acting on {target}."
        else:
            text = "I need your confirmation before performing that action."
        print(f"[AI] Approval required: {reason}", flush=True)
        self.event_bus.emit("assistant_response", text=text)

    def on_tool_result(self, result):
        if not isinstance(result, ToolResult):
            return
        text = self._format_tool_success(result) if result.success else self._format_tool_failure(result)
        self.event_bus.emit("assistant_response", text=text)

    @staticmethod
    def _format_tool_success(result: ToolResult) -> str:
        output = result.output
        if isinstance(output, dict):
            target = output.get("target")
            if result.tool == "system.open_application" and target:
                return f"Opened {target}."
            if result.tool == "system.launch_application" and target:
                return f"Launched {target}."
            if result.tool == "system.close_application" and target:
                return f"Closed {target}."
            if result.tool == "system.stop_process" and output.get("pid"):
                return f"Stopped process {output['pid']}."
            if result.tool == "system.start_process" and target:
                return f"Started {target}."
        if output is None:
            return f"{result.tool} completed successfully."
        return str(output)

    @staticmethod
    def _format_tool_failure(result: ToolResult) -> str:
        if result.error:
            return f"I couldn't complete that action: {result.error}"
        return "I couldn't complete that action."

    def _generate_response(self, text):
        def on_sentence(sentence):
            if sentence:
                self.event_bus.emit("assistant_sentence", text=sentence)

        response = self.engine.generate_response(text, on_sentence=on_sentence)
        if not response:
            print("[AI] No response generated.", flush=True)
            return
        self.event_bus.emit("assistant_response", text=response)
