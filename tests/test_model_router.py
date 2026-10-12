from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest

from ai.llm_provider import create_llm_provider
from ai.llama_cpp_engine import LlamaCppEngine
from ai.model_router import ModelRoutingError, ModelRuntime, RoutedLLMProvider
from core.event_bus import EventBus


@pytest.fixture(autouse=True)
def clean_router_environment(monkeypatch):
    monkeypatch.setenv("ASTA_MODEL_ROUTER", "0")
    monkeypatch.delenv("ASTA_LLM_MODEL", raising=False)
    for key in list(os.environ):
        if key.startswith("ASTA_CODING_"):
            monkeypatch.delenv(key, raising=False)


class FakeManager:
    def __init__(self, profile, log):
        self.profile = profile
        self.env_prefix = "ASTA_LLM" if profile == "general" else "ASTA_CODING_LLM"
        self.base_url = f"http://127.0.0.1:{8080 if profile == 'general' else 8081}/v1"
        self.model_path = f"/models/{profile}.gguf"
        self.log = log
        self.owned = False
        self.ready = False
        self.valid = True
        self.start_ok = True
        self.stop_ok = True

    @property
    def running(self):
        return self.owned and self.ready

    def _resolve_server_path(self):
        return "llama-server"

    def _resolve_model_path(self, server):
        return self.model_path if self.valid else None

    def _server_is_ready(self):
        return self.ready

    def ensure_running(self):
        if not self.ready:
            self.log.append("start:" + self.profile)
            self.ready = self.start_ok
            self.owned = self.start_ok
        return self.ready

    def stop(self):
        self.log.append("stop:" + self.profile)
        if not self.stop_ok:
            return False
        self.ready = False
        self.owned = False
        return True


class FakeEngine:
    def __init__(self, *, base_url, server_manager, env_prefix, max_output_tokens, reasoning_retry_tokens):
        self.server_manager = server_manager
        self.model = None
        self.system_prompt = "default"
        self.max_output_tokens = max_output_tokens
        self.reasoning_retry_tokens = reasoning_retry_tokens
        self.temperature = 0.7
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self.session = SimpleNamespace(close=lambda: None)
        self.cancelled = False
        self.seen = []

    def _discover_model(self):
        self.model = self.server_manager.model_path
        return self.model

    def reset_conversation(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.reset_conversation()

    def _trim_history(self):
        self._messages = self._messages[:1] + self._messages[1:][-24:]

    def generate_response(self, text, on_sentence=None, context=None):
        self.seen = [dict(item) for item in self._messages]
        answer = "generated: " + text
        self._messages += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        if on_sentence:
            on_sentence(answer)
        return answer

    def complete_json(self, system, user, max_tokens=96, timeout=None):
        return '{"action":"none"}'

    def warmup(self):
        return True

    def cancel(self):
        self.cancelled = True


def build_runtime():
    log = []
    kernel = SimpleNamespace(event_bus=EventBus())
    managers = {role: FakeManager(role, log) for role in ("general", "coding")}
    runtime = ModelRuntime(kernel, managers=managers)
    return runtime, managers, log


def client(runtime, purpose="chat"):
    return RoutedLLMProvider(runtime, purpose=purpose, engine_factory=FakeEngine)


@pytest.mark.parametrize("text,expected", [
    ("What is the capital of India?", "general"),
    ("Open Chrome", "general"),
    ("Tell me my postal code", "general"),
    ("Debug this Python traceback", "coding"),
    ("Refactor the repository and run pytest", "coding"),
    ("Add a function to src/main.py", "coding"),
    ("Explain this SQL query", "coding"),
])
def test_rules_route_user_goal_not_internal_prompt(text, expected):
    runtime, _, _ = build_runtime()
    assert runtime.select(text) == expected


def test_owned_models_switch_in_order_and_no_other_model_remains_running():
    runtime, managers, log = build_runtime()
    provider = client(runtime)
    assert provider.warmup()
    provider.generate_response("Debug Python code")
    assert managers["coding"].running
    assert not managers["general"].running
    provider.generate_response("What is the capital of India?")
    assert log == ["start:general", "stop:general", "start:coding", "stop:coding", "start:general"]


@pytest.mark.parametrize("failure", ["missing_file", "missing_config", "external_target"])
def test_target_preflight_preserves_working_general_model(failure):
    runtime, managers, log = build_runtime()
    provider = client(runtime)
    provider.warmup()
    if failure == "missing_file":
        managers["coding"].valid = False
    elif failure == "missing_config":
        managers["coding"].model_path = None
    else:
        managers["coding"].ready = True  # Not ASTA-owned.
    with pytest.raises(ModelRoutingError):
        provider.generate_for_task("prompt", goal="Debug Python")
    assert managers["general"].running
    assert "stop:general" not in log
    assert "start:coding" not in log


def test_external_general_server_is_not_stopped_to_load_coding():
    runtime, managers, log = build_runtime()
    managers["general"].ready = True
    provider = client(runtime)
    assert provider.warmup()  # Matching general endpoint can be reused.
    with pytest.raises(ModelRoutingError, match="externally owned"):
        provider.generate_for_task("prompt", goal="Debug Python")
    assert log == []
    runtime.shutdown()
    assert managers["general"].ready


def test_failed_termination_prevents_coding_launch():
    runtime, managers, log = build_runtime()
    provider = client(runtime)
    provider.warmup()
    managers["general"].stop_ok = False
    with pytest.raises(ModelRoutingError, match="termination"):
        provider.generate_for_task("prompt", goal="Debug Python")
    assert managers["general"].owned
    assert "start:coding" not in log


def test_coding_start_failure_is_visible_and_has_no_general_generation_fallback():
    runtime, managers, log = build_runtime()
    provider = client(runtime)
    provider.warmup()
    managers["coding"].start_ok = False
    sentences = []
    result = provider.generate_response("Debug Python", on_sentence=sentences.append)
    assert "No alternate model was used" in result
    assert sentences == [result]
    assert not provider.engines["general"].seen
    assert log == ["start:general", "stop:general", "start:coding"]
    provider.generate_response("Hello")
    assert managers["general"].running


def test_chat_history_crosses_profiles_but_agent_prompt_does_not_pollute_chat():
    runtime, _, _ = build_runtime()
    chat = client(runtime)
    brain = client(runtime, "agent")
    chat.set_system_prompt("chat system")
    brain.set_system_prompt("agent system")
    chat.generate_response("Hello")
    chat.generate_response("Explain Python")
    assert chat.engines["coding"].seen[1]["content"] == "Hello"
    brain.generate_for_task("control JSON prompt", goal="Debug Python")
    chat.generate_response("Tell me a joke")
    assert chat.engines["general"].seen[0]["content"] == "chat system"
    assert "control JSON prompt" not in str(chat.engines["general"].seen)
    assert brain.engines["coding"].max_output_tokens == 2048
    chat.reset_conversation()
    assert not chat.history


def test_task_route_is_sticky_even_for_ambiguous_decision_goal():
    runtime, _, _ = build_runtime()
    brain = client(runtime, "agent")
    brain.generate_for_task("write JSON", goal="Inspect this project", coding_scope=True)
    runtime.kernel.event_bus.emit("task_created", task={"id": "task-1", "goal": "Inspect this project"})
    assert runtime.select("continue", task_id="task-1") == "coding"
    brain.generate_for_task("next decision", goal="continue", task_id="task-1")
    assert runtime.active_profile == "coding"
    assert runtime.select("What is the weather?") == "general"


def test_followup_and_intent_json_reuse_coding_without_extra_switch():
    runtime, _, log = build_runtime()
    provider = client(runtime)
    provider.generate_response("Debug Python")
    provider.complete_json("classify", "fix that failure")
    provider.generate_response("Fix that failure")
    assert log == ["start:coding"]


def test_requests_are_serialized_and_cannot_unload_inflight_model():
    runtime, managers, log = build_runtime()
    provider = client(runtime)
    provider.warmup()
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    errors = []

    def hold(_engine):
        entered.set()
        assert release.wait(3)

    def first():
        try:
            runtime.execute("general", provider._engine("general"), hold)
        except Exception as exc:
            errors.append(exc)

    def second():
        second_started.set()
        try:
            provider.generate_for_task("prompt", goal="Debug Python")
        except Exception as exc:
            errors.append(exc)

    thread1 = threading.Thread(target=first)
    thread2 = threading.Thread(target=second)
    thread1.start()
    assert entered.wait(2)
    thread2.start()
    assert second_started.wait(2)
    assert managers["general"].running
    assert "stop:general" not in log
    release.set()
    thread1.join(3)
    thread2.join(3)
    assert not thread1.is_alive() and not thread2.is_alive()
    assert not errors
    assert runtime.active_profile == "coding"


def test_shutdown_cancels_clients_and_rejects_new_requests():
    runtime, managers, _ = build_runtime()
    provider = client(runtime)
    provider.warmup()
    runtime.shutdown()
    assert provider.engines["general"].cancelled
    assert not managers["general"].owned
    with pytest.raises(ModelRoutingError, match="shut down"):
        provider.generate_for_task("prompt", goal="Debug Python")


def test_engine_profile_settings_do_not_inherit_general_model(monkeypatch):
    monkeypatch.setenv("ASTA_LLM_MODEL", "nemotron")
    monkeypatch.setenv("ASTA_LLM_MAX_OUTPUT_TOKENS", "123")
    manager = FakeManager("coding", [])
    engine = LlamaCppEngine(base_url=manager.base_url, server_manager=manager,
                            env_prefix="ASTA_CODING_LLM", max_output_tokens=2048)
    assert engine.model is None
    assert engine.max_output_tokens == 2048
    assert engine.server_manager is manager


def test_factory_is_opt_in_and_shares_ownership_not_client_state(monkeypatch):
    kernel = SimpleNamespace(event_bus=EventBus())
    assert isinstance(create_llm_provider(), LlamaCppEngine)
    monkeypatch.setenv("ASTA_MODEL_ROUTER", "1")
    chat = create_llm_provider(kernel=kernel)
    brain = create_llm_provider(kernel=kernel, purpose="agent")
    assert isinstance(chat, RoutedLLMProvider)
    assert chat.runtime is brain.runtime
    assert chat.engines["general"] is not brain.engines["general"]
    assert chat.engines["general"].server_manager is brain.engines["general"].server_manager
    with pytest.raises(ValueError, match="kernel-scoped"):
        create_llm_provider()


def test_router_rejects_same_port_and_nonlocal_endpoints():
    runtime, managers, _ = build_runtime()
    managers["coding"].base_url = "http://localhost:8080/v1"
    with pytest.raises(ValueError, match="different"):
        ModelRuntime(runtime.kernel, managers=managers)
    managers["coding"].base_url = "http://remote.example:8081/v1"
    with pytest.raises(ValueError, match="loopback"):
        ModelRuntime(runtime.kernel, managers=managers)


def test_wrong_model_on_external_general_endpoint_is_rejected_without_killing_it():
    runtime, managers, log = build_runtime()
    managers["general"].ready = True
    provider = client(runtime)
    provider.engines["general"]._discover_model = lambda: "wrong.gguf"
    assert not provider.warmup()
    assert log == []
    assert managers["general"].ready


def test_agent_brain_passes_original_goal_and_task_id_to_router(monkeypatch):
    from core.agent.brain import AgentBrain, AgentDecision
    from core.contracts import IntentResult, IntentType
    calls = []
    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "{}"
    provider = SimpleNamespace(generate_for_task=generate)
    brain = AgentBrain(SimpleNamespace(), provider=provider, enabled=True)
    monkeypatch.setattr(brain, "_build_prompt", lambda *args: "internal capabilities include Python")
    monkeypatch.setattr(brain, "_parse_response", lambda text: "proposal")
    monkeypatch.setattr(brain, "_validate_proposal", lambda value: None)
    intent = IntentResult(intent=IntentType.TASK, confidence=1.0, normalized_text="Inspect this project")
    assert brain.plan("Inspect this project", intent=intent) == "proposal"
    assert calls[-1][1] == {"goal": "Inspect this project", "coding_scope": True}
    monkeypatch.setattr(brain, "_build_decision_prompt", lambda *args: "internal decision prompt")
    monkeypatch.setattr(brain, "_parse_decision", lambda text: AgentDecision())
    monkeypatch.setattr(brain, "_normalize_visual_next_action", lambda value, state: value)
    monkeypatch.setattr(brain, "_enforce_goal_verification_gate", lambda value, state: value)
    monkeypatch.setattr(brain, "_validate_decision", lambda value: None)
    brain.decide({"id": "task-1", "goal": "continue"})
    assert calls[-1][1] == {"goal": "continue", "task_id": "task-1"}


def test_kernel_shutdown_owns_shared_runtime_cleanup():
    from core import Kernel
    calls = []
    kernel = Kernel()
    kernel.model_runtime = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    kernel.shutdown()
    kernel.shutdown()
    assert calls == ["shutdown"]


def test_startup_rollback_cleans_shared_model_runtime():
    from core import Kernel
    calls = []
    kernel = Kernel()
    kernel.model_runtime = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    def fail():
        raise RuntimeError("module failed")
    kernel.register_module(SimpleNamespace(name="failing", initialize=fail))
    with pytest.raises(RuntimeError, match="module failed"):
        kernel.start()
    assert calls == ["shutdown"]


def test_coding_task_followup_routes_even_when_chat_client_did_not_generate_plan():
    runtime, _, log = build_runtime()
    brain = client(runtime, "agent")
    chat = client(runtime)
    brain.generate_for_task("plan", goal="Debug Python")
    chat.generate_response("Fix that failure")
    assert runtime.active_profile == "coding"
    assert log == ["start:coding"]


def test_json_classification_does_not_read_or_change_chat_history():
    runtime, _, _ = build_runtime()
    provider = client(runtime)
    provider.generate_response("Hello")
    history = [dict(item) for item in provider.history]
    provider.complete_json("classify", "test input")
    assert provider.history == history
