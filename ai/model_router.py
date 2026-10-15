"""Kernel-owned model services: safe switching or independent dual residency."""
from __future__ import annotations

import os
import re
import threading
import time
import weakref
from collections import OrderedDict
from urllib.parse import urlparse

from core import gpu_share
from .llama_cpp_engine import LlamaCppEngine
from .llama_server_manager import LlamaServerManager


class ModelRoutingError(RuntimeError):
    pass


_CODE = re.compile(
    r"\b(?:code|coding|debug(?:ging)?|refactor(?:ing)?|pytest|unittest|traceback|"
    r"python|javascript|typescript|sql|compiler|codebase|repository|"
    r"write\s+code|review\s+(?:the\s+)?code|fix\s+(?:the\s+)?(?:bug|test)|"
    r"(?:implement|write|add|create|modify|test)\s+(?:a\s+|the\s+)?(?:function|script|class|unit\s+test))\b"
    r"|\b[\w.-]+\.(?:py|js|ts|tsx|jsx|rs|cpp|java|go)\b|```",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(r"^(?:please\s+)?(?:fix|retry|try again|continue|resume|change that|explain that)\b", re.I)


def _name(value):
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


class ModelRuntime:
    """Shared ownership/serialization only; client prompts and histories stay separate."""

    def __init__(self, kernel, *, managers=None):
        self.kernel = kernel
        self.lock = threading.RLock()  # Metadata / single-resident lifecycle only.
        self.mode = os.getenv("ASTA_MODEL_MODE", "on_demand").strip().lower()
        if self.mode not in {"on_demand", "dual"}:
            raise ValueError("ASTA_MODEL_MODE must be on_demand or dual.")
        self.profile_locks = {p: threading.RLock() for p in ("general", "coding")}
        self.priority = threading.Condition()
        self.general_requests = 0
        self.closing = threading.Event()
        self.active_profile = None
        self.clients = weakref.WeakSet()
        self.task_profiles = OrderedDict()
        self.pending_goals = OrderedDict()
        if managers is None:
            general_url = os.getenv("ASTA_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
            coding_url = os.getenv("ASTA_CODING_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
            managers = {
                "general": LlamaServerManager(base_url=general_url),
                "coding": LlamaServerManager(base_url=coding_url, env_prefix="ASTA_CODING_LLM"),
            }
        self.managers = managers
        ports = []
        for manager in managers.values():
            url = urlparse(manager.base_url)
            if url.hostname not in {"127.0.0.1", "localhost", "::1"} or url.scheme != "http":
                raise ValueError("Model switching requires loopback HTTP llama.cpp endpoints.")
            ports.append(url.port or 80)
        if len(set(ports)) != len(ports):
            raise ValueError("General and coding models must use different local ports.")
        bus = getattr(kernel, "event_bus", None)
        if bus is not None:
            bus.subscribe("task_created", self.on_task_created)

    def emit(self, profile, status):
        model = _name(self.managers[profile].model_path)
        print(f"[ModelRouter] {profile}: {status} ({model or 'unconfigured'})", flush=True)
        bus = getattr(self.kernel, "event_bus", None)
        if bus is not None:
            bus.emit("model_route_changed", profile=profile, status=status, model=model)

    def select(self, goal, *, task_id=None, coding_scope=False, followup_profile=None):
        with self.lock:
            if task_id and task_id in self.task_profiles:
                return self.task_profiles[task_id]
            text = str(goal or "").strip()
            text = re.sub(
                r"\b(?:postal|zip|dress|morse|qr|verification|access)\s+codes?\b",
                "", text, flags=re.I,
            )
            profile = "coding" if (
                coding_scope or _CODE.search(text)
                or (_FOLLOWUP.match(text) and followup_profile == "coding")
            ) else "general"
            if task_id:
                self.task_profiles[task_id] = profile
                while len(self.task_profiles) > 128:
                    self.task_profiles.popitem(last=False)
            return profile

    def on_task_created(self, task=None, **kwargs):
        if not isinstance(task, dict) or not task.get("id"):
            return
        with self.lock:
            goal = str(task.get("goal") or "").strip()
            profile = self.pending_goals.pop(goal, None)
            if profile is None:
                profile = self.select(goal, task_id=task["id"])
            self.task_profiles[task["id"]] = profile
            while len(self.task_profiles) > 128:
                self.task_profiles.popitem(last=False)

    def _validate_files(self, profile):
        manager = self.managers[profile]
        # Explicit paths avoid discovering the wrong GGUF after a second download.
        if not manager.model_path:
            raise ModelRoutingError(f"Set {manager.env_prefix}_MODEL_PATH explicitly before routing.")
        server = manager._resolve_server_path()
        if not server or not manager._resolve_model_path(server):
            raise ModelRoutingError(f"The {profile} llama-server executable or GGUF file is unavailable.")

    def _activate(self, profile, engine):
        target = self.managers[profile]
        # Validate target BEFORE unloading a working general/coding server.
        self._validate_files(profile)
        if profile == "coding" and target._server_is_ready() and not (target.owned and target.running):
            if self.mode == "on_demand":
                raise ModelRoutingError("Coding endpoint is externally owned; use a free port for ASTA-managed coding.")
        for other_profile, other in (self.managers.items() if self.mode == "on_demand" else ()):
            if other_profile == profile:
                continue
            if not other.running and not other._server_is_ready():
                continue
            if not other.owned:
                raise ModelRoutingError(
                    f"Cannot switch: the {other_profile} endpoint is externally owned. "
                    "Stop it yourself or disable ASTA_MODEL_ROUTER; ASTA will not terminate it."
                )
            self.emit(profile, "loading")
            if not other.stop():
                raise ModelRoutingError(f"Could not confirm termination of the {other_profile} server.")
            self.active_profile = None
            deadline = time.monotonic() + 5
            while other.running or other._server_is_ready():
                if time.monotonic() >= deadline:
                    raise ModelRoutingError(f"The {other_profile} endpoint is still occupied after stopping.")
                time.sleep(0.1)
            self.emit(other_profile, "stopped")

        already_ready = target._server_is_ready()
        if profile != self.active_profile or not already_ready:
            self.emit(profile, "loading")
        gpu_share.release_idle(f"loading the {profile} model")
        if not target.ensure_running():
            raise ModelRoutingError(f"The {profile} model did not become ready; check llama.cpp and VRAM.")
        if self.closing.is_set():
            raise ModelRoutingError("Model runtime is shutting down.")
        try:
            model = engine._discover_model()
        except Exception as exc:
            if target.owned:
                target.stop()
            raise ModelRoutingError(f"Could not verify the {profile} endpoint's model identity.") from exc
        expected = os.getenv(f"{target.env_prefix}_MODEL") or target.model_path
        if _name(model).casefold() != _name(expected).casefold():
            if target.owned:
                target.stop()
            raise ModelRoutingError(f"The {profile} endpoint reported a different model than configured.")
        self.active_profile = profile
        self.emit(profile, "ready")

    def execute(self, profile, engine, operation):
        # Separate services may overlap; one request per profile remains serialized.
        lock = self.lock if self.mode == "on_demand" else self.profile_locks[profile]
        foreground = self.mode == "dual" and profile == "general"
        if foreground:
            with self.priority:
                self.general_requests += 1
        try:
            if self.mode == "dual" and profile == "coding":
                with self.priority:
                    while self.general_requests and not self.closing.is_set():
                        self.priority.wait(0.1)
            return self._execute_locked(profile, engine, operation, lock)
        finally:
            if foreground:
                with self.priority:
                    self.general_requests -= 1
                    self.priority.notify_all()

    def _execute_locked(self, profile, engine, operation, lock):
        with lock:
            if self.closing.is_set():
                raise ModelRoutingError("Model runtime is shut down.")
            try:
                self._activate(profile, engine)
                self.emit(profile, "generating")
                result = operation(engine)
                self.emit(profile, "ready")
                return result
            except ModelRoutingError:
                self.emit(profile, "error")
                raise
            except Exception as exc:
                self.emit(profile, "error")
                raise ModelRoutingError(
                    f"The {profile} model request failed ({type(exc).__name__})."
                ) from exc

    def shutdown(self):
        self.closing.set()
        with self.priority:
            self.priority.notify_all()
        for client in tuple(self.clients):
            client.cancel()
        with self.lock:
            for profile, manager in self.managers.items():
                with self.profile_locks[profile]:
                    if manager.owned and not manager.stop():
                        print("[ModelRouter] Owned server did not stop; retaining its process handle.", flush=True)
            self.active_profile = None
            bus = getattr(self.kernel, "event_bus", None)
            if bus is not None:
                bus.unsubscribe("task_created", self.on_task_created)
            for client in tuple(self.clients):
                for engine in client.engines.values():
                    engine.session.close()


class RoutedLLMProvider:
    def __init__(self, runtime, *, purpose="chat", engine_factory=LlamaCppEngine):
        self.runtime = runtime
        self.purpose = purpose
        self.engine_factory = engine_factory
        self.client_lock = threading.RLock()
        self.engines = {}
        general = self._engine("general")
        self.system_prompt = general.system_prompt
        self.max_output_tokens = general.max_output_tokens
        self.reasoning_retry_tokens = general.reasoning_retry_tokens
        self.temperature = general.temperature
        self.history = []
        self.last_profile = "general"
        self.runtime.clients.add(self)

    @property
    def model(self):
        engine = self.engines.get(self.last_profile)
        return engine.model if engine is not None else None

    def _engine(self, profile):
        if profile not in self.engines:
            manager = self.runtime.managers[profile]
            prefix = manager.env_prefix
            engine = self.engine_factory(
                base_url=manager.base_url, server_manager=manager, env_prefix=prefix,
                max_output_tokens=2048 if profile == "coding" else 256,
                reasoning_retry_tokens=2048 if profile == "coding" else 512,
            )
            if hasattr(self, "system_prompt"):
                engine.set_system_prompt(self.system_prompt)
            self.engines[profile] = engine
        return self.engines[profile]

    def set_system_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("System prompt must be non-empty.")
        with self.client_lock:
            self.system_prompt = prompt.strip()
            for engine in self.engines.values():
                engine.set_system_prompt(self.system_prompt)
            self.history = []

    def reset_conversation(self):
        with self.client_lock:
            self.history = []
            self.last_profile = "general"
            for engine in self.engines.values():
                engine.reset_conversation()

    def cancel(self):
        for engine in tuple(self.engines.values()):
            engine.cancel()

    def shutdown(self):
        # Client lifetime is not shared process lifetime. Kernel owns the runtime.
        self.cancel()

    def warmup(self):
        try:
            return self.runtime.execute("general", self._engine("general"), lambda engine: engine.warmup())
        except Exception as exc:
            print(f"[ModelRouter] General warmup unavailable: {exc}", flush=True)
            return False

    def _configure(self, engine, profile):
        if self.purpose == "agent":
            budget = int(os.getenv("ASTA_CODING_AGENT_MAX_OUTPUT_TOKENS", "4096")) if profile == "coding" else self.max_output_tokens
            engine.max_output_tokens = max(64, budget)
            engine.reasoning_retry_tokens = engine.max_output_tokens
            engine.temperature = float(os.getenv("ASTA_CODING_AGENT_TEMPERATURE", "0.0")) if profile == "coding" else self.temperature
        elif profile == "general":
            engine.max_output_tokens = self.max_output_tokens
            engine.reasoning_retry_tokens = self.reasoning_retry_tokens
            engine.temperature = self.temperature
        else:
            engine.temperature = float(os.getenv("ASTA_CODING_LLM_TEMPERATURE", "0.6"))

    def generate_for_task(self, prompt, *, goal, task_id=None, coding_scope=False):
        with self.client_lock:
            profile = self.runtime.select(goal, task_id=task_id, coding_scope=coding_scope)
            engine = self._engine(profile)
        def operation(e):
            with self.client_lock:
                self._configure(e, profile)
                e.reset_conversation()
                self.last_profile = profile
                response = e.generate_response(prompt, context=None)
                outcome = getattr(e, "last_generation", None) or {}
                if outcome.get("cancelled") or outcome.get("finish_reason") == "length":
                    reason = "timeout" if outcome.get("timed_out") else ("cancelled" if outcome.get("cancelled") else "output limit")
                    raise ModelRoutingError(f"Agent generation incomplete ({reason}); no partial plan will execute.")
                return response
        response = self.runtime.execute(profile, engine, operation)
        if not task_id and response:
            with self.runtime.lock:
                self.runtime.pending_goals[str(goal).strip()] = profile
                while len(self.runtime.pending_goals) > 64:
                    self.runtime.pending_goals.popitem(last=False)
        return response

    def generate_response(self, text, on_sentence=None, context=None):
        if not text:
            return ""
        with self.client_lock:
            manager = getattr(self.runtime.kernel, "task_manager", None)
            task = manager.current() if manager is not None else None
            task_profile = self.runtime.task_profiles.get(getattr(task, "id", None))
            followup = task_profile or ("coding" if self.runtime.active_profile == "coding" else self.last_profile)
            profile = "general" if self.runtime.mode == "dual" else self.runtime.select(text, followup_profile=followup)
            engine = self._engine(profile)
        def operation(e):
            with self.client_lock:
                self._configure(e, profile)
                e._messages = [{"role": "system", "content": self.system_prompt}] + [dict(item) for item in self.history]
                e._trim_history()
                response = e.generate_response(text, on_sentence=on_sentence, context=context)
                self.history = [dict(item) for item in e._messages[1:]]
                self.last_profile = profile
                return response
        try:
            return self.runtime.execute(profile, engine, operation)
        except ModelRoutingError as exc:
            response = f"I couldn't use the requested {profile} model: {exc} No alternate model was used."
            if on_sentence:
                on_sentence(response)
            return response

    def complete_json(self, system, user, max_tokens=96, timeout=None):
        with self.client_lock:
            profile = "general" if self.runtime.mode == "dual" else (self.runtime.active_profile or "general")
            engine = self._engine(profile)
        return self.runtime.execute(profile, engine, lambda e: e.complete_json(system, user, max_tokens=max_tokens, timeout=timeout))
