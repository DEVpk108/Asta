import json
import time
from urllib.parse import urlparse

import requests


class AIEngine:

    def __init__(
        self,
        base_url="http://127.0.0.1:1234",
        model="nvidia/nemotron-3-nano-4b",
        timeout=120,
        max_output_tokens=256,
        reasoning_retry_tokens=512,
        reasoning="off",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.reasoning_retry_tokens = reasoning_retry_tokens
        self.reasoning = reasoning
        self.session = requests.Session()

        # A.S.T.A. talks to LM Studio locally. Bypass environment proxies for
        # loopback traffic so local requests go directly to LM Studio.
        host = urlparse(self.base_url).hostname
        if host in {"localhost", "127.0.0.1", "::1"}:
            self.session.trust_env = False

        self.chat_url = f"{self.base_url}/api/v1/chat"
        self.models_url = f"{self.base_url}/api/v1/models"
        self.load_url = f"{self.base_url}/api/v1/models/load"
        self.previous_response_id = None
        self.model_instance_id = None
        self.warmed = False

        self.system_prompt = (
            "You are ASTA, a local AI voice assistant. "
            "Respond naturally and concisely. "
            "Prefer 1–3 short sentences for normal questions. "
            "Avoid long explanations unless the user asks for detail. "
            "Sound conversational, helpful, and direct."
        )

    def set_system_prompt(self, prompt):
        """Replace the grounding prompt used when starting a new conversation."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("System prompt must be a non-empty string.")
        self.system_prompt = prompt.strip()
        # A changed system prompt must start a fresh LM Studio response chain;
        # otherwise the server would continue the old conversation context.
        self.previous_response_id = None

    def _find_loaded_instance(self):
        """Return an already-loaded instance id for the selected model.

        LM Studio exposes loaded instances through GET /api/v1/models. We must
        check this before calling /models/load because an explicit load creates
        another model instance instead of simply reusing an existing one.
        """
        response = self.session.get(self.models_url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        for model_info in payload.get("models", []):
            if model_info.get("key") != self.model:
                continue
            for instance in model_info.get("loaded_instances", []) or []:
                instance_id = instance.get("id")
                if instance_id:
                    return instance_id
        return None

    def warmup(self):
        """Reuse/load the selected model and run a tiny throwaway generation.

        Existing LM Studio model instances are reused. A new instance is
        explicitly loaded only when the selected model has no loaded instance.
        If model listing is temporarily unavailable, the chat warm-up is still
        allowed to proceed because /api/v1/chat automatically loads the model
        when necessary.
        """
        start = time.perf_counter()
        try:
            instance_id = None
            try:
                list_start = time.perf_counter()
                instance_id = self._find_loaded_instance()
                print(
                    f"[AI] Loaded-model check: {time.perf_counter() - list_start:.3f}s",
                    flush=True,
                )
            except requests.RequestException as exc:
                print(
                    f"[AI] Loaded-model check unavailable: {type(exc).__name__}: {exc}",
                    flush=True,
                )

            if instance_id:
                self.model_instance_id = instance_id
                print(
                    f"[AI] Reusing loaded model instance: {instance_id}",
                    flush=True,
                )
            else:
                load_start = time.perf_counter()
                response = self.session.post(
                    self.load_url,
                    json={"model": self.model},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                try:
                    load_result = response.json()
                except ValueError:
                    load_result = {}
                self.model_instance_id = load_result.get("instance_id")
                print(
                    f"[AI] Model load request: {time.perf_counter() - load_start:.3f}s",
                    flush=True,
                )
                if self.model_instance_id:
                    print(
                        f"[AI] Loaded model instance: {self.model_instance_id}",
                        flush=True,
                    )

            warmup_start = time.perf_counter()
            response = self.session.post(
                self.chat_url,
                json={
                    "model": self.model,
                    "input": "Ready.",
                    "stream": False,
                    "store": False,
                    "max_output_tokens": 1,
                    "reasoning": "off",
                },
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            print(
                f"[AI] Inference warm-up: {time.perf_counter() - warmup_start:.3f}s",
                flush=True,
            )
            self.warmed = True
            print(
                f"[AI] Warm-up complete: {time.perf_counter() - start:.3f}s",
                flush=True,
            )
            return True
        except requests.RequestException as exc:
            print(f"[AI] Warm-up unavailable: {type(exc).__name__}: {exc}", flush=True)
            return False
        except Exception as exc:
            print(f"[AI] Warm-up error: {type(exc).__name__}: {exc}", flush=True)
            return False

    def reset_conversation(self):
        self.previous_response_id = None
        print("[AI] Conversation memory reset.", flush=True)

    @staticmethod
    def _extract_message_text(output):
        if isinstance(output, str):
            return output.strip()
        if not isinstance(output, list):
            return ""
        parts = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _output_stats(final_result):
        stats = final_result.get("stats", {}) if final_result else {}
        return (
            stats.get("input_tokens", 0),
            stats.get("total_output_tokens", 0),
            stats.get("reasoning_output_tokens", 0),
            stats.get("tokens_per_second", 0.0),
            stats.get("time_to_first_token_seconds"),
            stats.get("model_load_time_seconds"),
        )

    @staticmethod
    def _normalize_text(text):
        if not isinstance(text, str) or ("\u00c3" not in text and "\u00e2" not in text):
            return text
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        return repaired if repaired != text else text

    @staticmethod
    def _is_speech_worthy(text):
        """Keep emoji/punctuation-only fragments out of the TTS queue."""
        return bool(text and any(char.isalnum() for char in text))

    def _request(self, text, on_sentence, max_output_tokens):
        payload = {
            "model": self.model,
            "input": text,
            "stream": True,
            "store": True,
            "max_output_tokens": max_output_tokens,
            "reasoning": self.reasoning,
        }

        if self.previous_response_id is None:
            payload["system_prompt"] = self.system_prompt
        else:
            payload["previous_response_id"] = self.previous_response_id

        request_start = time.perf_counter()
        response_open = None
        first_event_time = None
        chat_start_time = None
        model_load_start = None
        model_load_end = None
        prompt_start = None
        prompt_end = None
        message_start = None
        first_delta_time = None
        full_text = ""
        sentence_buffer = ""
        final_result = None

        with self.session.post(
            self.chat_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response_open = time.perf_counter()
            response.raise_for_status()
            response.encoding = "utf-8"
            event_type = None

            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("event:"):
                    event_type = raw_line[len("event:"):].strip()
                    continue
                if not raw_line.startswith("data:"):
                    continue

                data_text = raw_line[len("data:"):].strip()
                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                if first_event_time is None:
                    first_event_time = now

                event_name = data.get("type") or event_type
