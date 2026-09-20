import json
import os
import re
import time
from urllib.parse import urlparse

import requests


class LlamaCppEngine:
    def __init__(
        self,
        base_url=None,
        model=None,
        timeout=120,
        max_output_tokens=256,
        reasoning_retry_tokens=512,
        reasoning="off",
    ):
        self.base_url = (base_url or os.getenv(
            "ASTA_LLM_BASE_URL", "http://127.0.0.1:8080/v1"
        )).rstrip("/")
        self.model = model or os.getenv("ASTA_LLM_MODEL") or None
        self.timeout = float(os.getenv("ASTA_LLM_TIMEOUT", timeout))
        self.max_output_tokens = int(os.getenv(
            "ASTA_LLM_MAX_OUTPUT_TOKENS", max_output_tokens
        ))
        self.reasoning_retry_tokens = int(os.getenv(
            "ASTA_LLM_REASONING_RETRY_TOKENS", reasoning_retry_tokens
        ))
        self.reasoning = reasoning
        self.session = requests.Session()

        host = urlparse(self.base_url).hostname
        if host in {"localhost", "127.0.0.1", "::1"}:
            self.session.trust_env = False

        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.previous_response_id = None
        self.warmed = False

        self.system_prompt = """
You are A.S.T.A. (Adaptive System for Technical Assistance), a professional local-first AI engineering assistant.

IDENTITY
You are a capable AI assistant designed to help the user understand, build, debug, learn, research, and solve problems. Your purpose is to augment the user's capabilities and decision-making, not replace them.

You are A.S.T.A., not ChatGPT. Do not claim capabilities, tools, sensors, memory, access, or permissions that have not actually been provided to you.

CORE BEHAVIOR
Be useful, truthful, technically competent, context-aware, practical, and professional.

Understand the user's intent before responding. Answer the actual question rather than a nearby question. Use available conversation context to maintain continuity.

Be proactive when it provides genuine value. Point out important risks, trade-offs, mistakes, missing considerations, or better approaches when relevant. Do not add suggestions merely to appear helpful.

COMMUNICATION
Speak naturally and directly.

For normal conversational questions, prefer concise responses. Use more detail when the problem is complex or the user asks for an explanation, tutorial, comparison, implementation, or deeper reasoning.

Avoid:
- unnecessary repetition
- generic filler
- excessive disclaimers
- overly formal or robotic language
- pretending to know something that is unavailable

When teaching, explain the underlying concept instead of only giving the final answer.

CONTEXT AND CONVERSATION
Treat the current conversation as the primary source of immediate context.

Use recent messages to understand references such as "this", "that", "it", "we", "earlier", and "what were we talking about".

Prefer the most recent relevant topic over unrelated older topics.

Do not invent missing conversation history. When required information is genuinely unavailable, state that clearly and continue with the best possible answer.

ENGINEERING MODE
You are especially strong as an engineering assistant for software development, AI/ML, LLMs, agentic systems, electronics, automation, debugging, and technical problem solving.

When solving technical problems:
1. Understand the actual problem and constraints.
2. Identify the relevant technical considerations.
3. Compare practical approaches when multiple solutions exist.
4. Recommend an appropriate solution.
5. Explain important trade-offs.
6. Provide implementation details when useful.

For software and code:
- Prefer correct, maintainable solutions over unnecessarily clever ones.
- Preserve the existing architecture unless a change is justified.
- Make the smallest reasonable change when modifying an existing system.
- Do not invent files, APIs, functions, dependencies, or project behavior.
- Consider failure cases and edge cases.
- Clearly distinguish assumptions from known facts.

For electronics and physical systems, consider feasibility, hardware constraints, safety, signal integrity, power, cost, and practical implementation details where relevant.

DECISION MAKING
When comparing solutions, consider reliability, complexity, performance, maintainability, cost, scalability, compatibility, and implementation effort.

For personal projects, prefer incremental and testable solutions. Avoid unnecessary over-engineering while keeping future expansion in mind.

UNCERTAINTY AND TRUTHFULNESS
Never fabricate facts, results, actions, tool usage, or capabilities.

Distinguish between known information, reasonable inference, and uncertainty. If an assumption is necessary, state it briefly.

When you do not know something, say so rather than creating a confident-sounding answer.

USER AGENCY
The user remains the final decision-maker. Give recommendations with reasoning, but do not present personal preferences as mandatory requirements unless a genuine technical or safety constraint makes them necessary.

PROJECT AWARENESS
The user is developing A.S.T.A. as a modular, local-first AI engineering assistant intended to evolve into a broader personal AI operating system.

The broader architecture may eventually include persistent memory, project awareness, workspaces, workflow management, proactive assistance, tool execution, specialized agents, knowledge retrieval, code execution and testing, and reflection/improvement loops.

These capabilities may not exist in the current version. Never claim that a future capability is already available.

IMPORTANT: Chat history and future long-term memory are different concepts. Conversation history provides immediate continuity; a future Memory Layer may provide durable knowledge and user/project context.

RESPONSE PRINCIPLE
Your goal is not merely to produce an answer. Help the user understand the problem, make better technical decisions, and build things effectively.
""".strip()
        self._messages = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("System prompt must be a non-empty string.")
        self.system_prompt = prompt.strip()
        self.reset_conversation()

    def _discover_model(self):
        response = self.session.get(self.models_url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", [])
        if not models:
            raise RuntimeError("llama-server returned no models.")

        if self.model:
            for item in models:
                if item.get("id") == self.model:
                    return self.model
            raise RuntimeError(
                f"Configured llama.cpp model '{self.model}' was not reported by the server."
            )

        model_id = models[0].get("id")
        if not model_id:
            raise RuntimeError("llama-server returned a model without an id.")
        self.model = model_id
        return model_id

    def warmup(self):
        start = time.perf_counter()
        try:
            model = self._discover_model()
            print(f"[AI] llama.cpp model: {model}", flush=True)
            response = self.session.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "Ready."},
                    ],
                    "stream": False,
                    "max_tokens": 1,
                    "temperature": 0,
                    "cache_prompt": True,
                },
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            self.warmed = True
            print(
                f"[AI] llama.cpp warm-up complete: "
                f"{time.perf_counter() - start:.3f}s",
                flush=True,
            )
            return True
        except requests.RequestException as exc:
            print(
                f"[AI] llama.cpp warm-up unavailable: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        except Exception as exc:
            print(
                f"[AI] llama.cpp warm-up error: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

    def reset_conversation(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self.previous_response_id = None
        print("[AI] Conversation memory reset.", flush=True)

    @staticmethod
    def _normalize_text(text):
        if not isinstance(text, str):
            return text
        if "Ã" not in text and "â" not in text:
            return text
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        return repaired if repaired != text else text

    @staticmethod
    def _is_speech_worthy(text):
        return bool(text and any(char.isalnum() for char in text))

    _NON_TERMINAL_ABBREVIATIONS = {
        "a.m", "p.m", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
        "st", "vs", "etc", "e.g", "i.e", "no", "fig", "approx",
    }

    @classmethod
    def _is_sentence_boundary(cls, buffer, punctuation_index):
        if punctuation_index >= len(buffer):
            return False

        punctuation = buffer[punctuation_index]
        if punctuation not in ".!?":
            return False

        next_index = punctuation_index + 1
        if next_index < len(buffer) and not buffer[next_index].isspace():
            return False

        prefix = buffer[:punctuation_index + 1]
        match = re.search(r"([^\s]+)$", prefix)
        token = match.group(1) if match else ""
        bare_token = token.rstrip(".").lower()

        if punctuation == ".":
            if bare_token in cls._NON_TERMINAL_ABBREVIATIONS:
                return False
            # Avoid splitting acronyms such as A.S.T.A. or U.S. when the
            # following word continues the same sentence. If the next word is
            # capitalized, treat the acronym period as a real sentence boundary.
            if re.fullmatch(r"(?:[A-Za-z]\.){2,}[A-Za-z]?\.?", token):
                remainder = buffer[punctuation_index + 1:]
                next_nonspace = re.search(r"\S", remainder)
                if next_nonspace is None:
                    return True
                return not next_nonspace.group(0).islower()

        return True

    @classmethod
    def _emit_sentence_chunks(cls, buffer):
        while True:
            sentence_end = None
            for match in re.finditer(r"[.!?](?=\s|$)", buffer):
                if cls._is_sentence_boundary(buffer, match.start()):
                    sentence_end = match.start()
                    break

            if sentence_end is None:
                return buffer, None

            sentence = buffer[:sentence_end + 1].strip()
            buffer = buffer[sentence_end + 1:]
            if not buffer.strip():
                buffer = ""
            if sentence:
                return buffer, sentence

    def _request(self, text, on_sentence, max_output_tokens, context=None):
        if self.model is None:
            self._discover_model()

        messages = list(self._messages)
        current_turn = context.strip() if isinstance(context, str) and context.strip() else text
        messages.append({"role": "user", "content": current_turn})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_output_tokens,
            "temperature": 0.7,
            "cache_prompt": True,
        }

        request_start = time.perf_counter()
        response_open = None
        first_event_time = None
        first_delta_time = None
        full_text = ""
        sentence_buffer = ""
        final_result = None
        final_id = None
        server_usage = {}
        server_timings = {}

        with self.session.post(
            self.chat_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        ) as response:
            response_open = time.perf_counter()
            response.raise_for_status()
            response.encoding = "utf-8"

            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue

                data_text = raw_line[len("data:"):].strip()
                if data_text == "[DONE]":
                    continue

                try:
                    data = json.loads(data_text)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                if first_event_time is None:
                    first_event_time = now

                final_id = data.get("id") or final_id
                choices = data.get("choices") or []
                choice = choices[0] if choices else {}
                delta = choice.get("delta") or {}
                content = self._normalize_text(delta.get("content", ""))

                if content:
                    if first_delta_time is None:
                        first_delta_time = now

                    full_text += content
                    sentence_buffer += content

                    while True:
                        sentence_buffer, sentence = self._emit_sentence_chunks(
                            sentence_buffer
                        )
                        if sentence is None:
                            break
                        sentence = self._normalize_text(sentence)
                        if on_sentence and self._is_speech_worthy(sentence):
                            on_sentence(sentence)

                if data.get("usage"):
                    server_usage = data["usage"]
                if data.get("timings"):
                    server_timings = data["timings"]
                if data.get("usage") or data.get("timings"):
                    final_result = data

        if not final_result:
            final_result = {}

        remaining = self._normalize_text(sentence_buffer.strip())
        if remaining and on_sentence and self._is_speech_worthy(remaining):
            on_sentence(remaining)

        usage = server_usage or final_result.get("usage") or {}
        timings = server_timings or final_result.get("timings") or {}

        prompt_n = int(timings.get("prompt_n") or 0)
        usage_details = usage.get("prompt_tokens_details") or {}
        cache_n = int(
            timings.get("cache_n")
            or usage_details.get("cached_tokens")
            or 0
        )
        predicted_n = int(
            timings.get("predicted_n")
            or usage.get("completion_tokens")
            or 0
        )
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predicted_ms = float(timings.get("predicted_ms") or 0.0)
        prompt_tps = float(
            timings.get("prompt_per_second")
            or (prompt_n / (prompt_ms / 1000.0) if prompt_n and prompt_ms else 0.0)
        )
        predicted_tps = float(
            timings.get("predicted_per_second")
            or (predicted_n / (predicted_ms / 1000.0) if predicted_n and predicted_ms else 0.0)
        )
        context_tokens = prompt_n + cache_n + predicted_n
        input_tokens = int(usage.get("prompt_tokens") or (prompt_n + cache_n))
        output_tokens = int(usage.get("completion_tokens") or predicted_n)
        client_first_content = (
            first_delta_time - request_start
            if first_delta_time is not None
            else None
        )

        result = {
            "id": final_id,
            "text": self._normalize_text(full_text.strip()),
            "result": final_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "tokens_per_second": predicted_tps,
            "ttft": client_first_content,
            "client_first_content_seconds": client_first_content,
            "prompt_tokens": prompt_n,
            "cached_tokens": cache_n,
            "predicted_tokens": predicted_n,
            "context_tokens": context_tokens,
            "prompt_ms": prompt_ms,
            "prompt_tokens_per_second": prompt_tps,
            "predicted_ms": predicted_ms,
            "predicted_tokens_per_second": predicted_tps,
            "server_timings": dict(timings),
            "server_usage": dict(usage),
            "model_load_time": None,
            "request_time": time.perf_counter() - request_start,
            "response_open_time": (
                response_open - request_start if response_open is not None else None
            ),
            "first_event_time": (
                first_event_time - request_start
                if first_event_time is not None
                else None
            ),
            "request_to_first_delta": client_first_content,
            "exhausted_reasoning": (
                not full_text.strip() and output_tokens >= max_output_tokens
            ),
        }

        self._messages.append({"role": "user", "content": text})
        self._messages.append(
            {"role": "assistant", "content": result["text"]}
        )
        return result

    def generate_response(self, text, on_sentence=None, context=None):
        if not text:
            return ""

        request_start = time.perf_counter()
        try:
            attempt = self._request(
                text,
                on_sentence,
                self.max_output_tokens,
                context=context,
            )
            if (
                attempt["exhausted_reasoning"]
                and self.reasoning_retry_tokens > self.max_output_tokens
            ):
                print(
                    f"[AI] Output budget exhausted; retrying with "
                    f"{self.reasoning_retry_tokens} output tokens.",
                    flush=True,
                )
                if len(self._messages) >= 3:
                    del self._messages[-2:]
                attempt = self._request(
                    text,
                    on_sentence,
                    self.reasoning_retry_tokens,
                    context=context,
                )
        except requests.RequestException as exc:
            print(
                f"[AI] llama.cpp connection error after "
                f"{time.perf_counter() - request_start:.2f}s: {exc}",
                flush=True,
            )
            return ""
        except Exception as exc:
            print(
                f"[AI] llama.cpp error after "
                f"{time.perf_counter() - request_start:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""

        print(f"[AI] Request: {attempt['request_time']:.2f}s", flush=True)
        print(
            f"[AI] Context: cached_input={attempt['cached_tokens']} "
            f"new_input={attempt['prompt_tokens']} "
            f"total_input={attempt['input_tokens']} "
            f"output={attempt['output_tokens']} "
            f"total={attempt['context_tokens']}",
            flush=True,
        )
        print(
            f"[AI] Server prompt: {attempt['prompt_ms'] / 1000.0:.3f}s "
            f"({attempt['prompt_tokens_per_second']:.2f} tok/s)",
            flush=True,
        )
        print(
            f"[AI] Server generation: {attempt['predicted_ms'] / 1000.0:.3f}s "
            f"({attempt['predicted_tokens_per_second']:.2f} tok/s)",
            flush=True,
        )
        print(f"[AI] Input tokens: {attempt['input_tokens']}", flush=True)
        print(f"[AI] Output tokens: {attempt['output_tokens']}", flush=True)
        if attempt["client_first_content_seconds"] is not None:
            print(
                f"[AI] Client first content: "
                f"{attempt['client_first_content_seconds']:.3f}s",
                flush=True,
            )
        if attempt["response_open_time"] is not None:
            print(
                f"[AI] HTTP headers received: "
                f"{attempt['response_open_time']:.3f}s",
                flush=True,
            )
        if attempt["first_event_time"] is not None and attempt["response_open_time"] is not None:
            print(
                f"[AI] Headers -> first SSE event: "
                f"{attempt['first_event_time'] - attempt['response_open_time']:.3f}s",
                flush=True,
            )
        if attempt.get("tokens_per_second"):
            print(
                f"[AI] Server generation speed: "
                f"{attempt['tokens_per_second']:.2f} tok/s",
                flush=True,
            )
        print(f"[AI] Response: {attempt['text']}", flush=True)
        return attempt["text"]
