import json
import os
import re
import threading
import time
from urllib.parse import urlparse

import requests

from core import gpu_share

from .llama_server_manager import LlamaServerManager


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
        self.temperature = float(os.getenv("ASTA_LLM_TEMPERATURE", "0.7"))
        self.reasoning = reasoning
        self.session = requests.Session()
        self.server_manager = LlamaServerManager(base_url=self.base_url)

        host = urlparse(self.base_url).hostname
        if host in {"localhost", "127.0.0.1", "::1"}:
            self.session.trust_env = False

        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.previous_response_id = None
        self.warmed = False
        # Set by cancel() (e.g. on a spoken barge-in) to stop the active stream.
        self._cancel_event = threading.Event()
        # Bound the replayed conversation so long sessions never exceed the
        # server context window. Counted in user+assistant turn pairs.
        try:
            self.max_history_turns = max(
                0, int(os.getenv("ASTA_LLM_HISTORY_TURNS", "12"))
            )
        except (TypeError, ValueError):
            self.max_history_turns = 12
        try:
            self.max_history_chars = max(
                0, int(os.getenv("ASTA_LLM_HISTORY_CHARS", "12000"))
            )
        except (TypeError, ValueError):
            self.max_history_chars = 12000

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

VOICE INPUT
User messages usually come from speech recognition and can contain misheard words, a stray wake word, or Hindi written phonetically. Interpret obvious mishearings from context. If a message is garbled or makes no sense, briefly ask the user to repeat it instead of guessing; never invent products, projects, or facts from an unfamiliar word.
Replies are spoken aloud: never use emoji. If the user speaks Hindi or Hinglish, reply in short, simple, grammatical Hindi (or Hinglish); if you are unsure what they meant, ask them to repeat.

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
            if not self.server_manager.ensure_running():
                print(
                    "[AI] llama.cpp server is not ready; "
                    "LLM warm-up skipped.",
                    flush=True,
                )
                return False

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

    def shutdown(self) -> None:
        self.server_manager.stop()
        self.warmed = False
        print("[AI] llama.cpp engine stopped.", flush=True)

    def cancel(self):
        """Stop the in-flight generation, if any, at the next stream chunk."""
        self._cancel_event.set()

    def _trim_history(self):
        """Keep the system prompt plus the most recent bounded turns."""
        if not self._messages:
            return
        system = self._messages[:1]
        turns = self._messages[1:]

        if self.max_history_turns >= 0:
            turns = turns[-(self.max_history_turns * 2):] if self.max_history_turns else []

        if self.max_history_chars:
            total = 0
            kept = []
            for message in reversed(turns):
                total += len(str(message.get("content") or ""))
                if total > self.max_history_chars and kept:
                    break
                kept.append(message)
            turns = list(reversed(kept))

        # Never start the replayed history with an orphaned assistant turn.
        while turns and turns[0].get("role") != "user":
            turns = turns[1:]

        self._messages = system + turns

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
        if punctuation not in ".!?\u0964\u0965":
            return False
        if punctuation in "\u0964\u0965":
            # Hindi danda / double danda always ends a sentence.
            next_index = punctuation_index + 1
            return next_index >= len(buffer) or buffer[next_index].isspace()

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
            if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?\.?", token):
                remainder = buffer[punctuation_index + 1:]
                next_nonspace = re.search(r"\S", remainder)
                if next_nonspace is None:
                    # Streaming: the next word has not arrived yet. Wait for
                    # it instead of speaking "A.S.T.A." as its own sentence;
                    # the end-of-stream flush still emits a trailing acronym.
                    return False
                return not next_nonspace.group(0).islower()

        return True

    @classmethod
    def _emit_sentence_chunks(cls, buffer):
        while True:
            sentence_end = None
            for match in re.finditer(r"[.!?\u0964\u0965](?=\s|$)", buffer):
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
            "temperature": self.temperature,
            "cache_prompt": True,
        }

        # Free VRAM held by an idle vision model before generating.
        gpu_share.release_idle("the chat reply")
        try:
            max_reply_seconds = float(os.getenv("ASTA_LLM_MAX_REPLY_SECONDS", "45"))
        except ValueError:
            max_reply_seconds = 45.0

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
        cancelled = False

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

            # chunk_size=None yields each HTTP chunk as soon as it arrives.
            # llama-server streams with chunked transfer encoding, so this is
            # just as responsive as the previous chunk_size=1 without doing a
            # Python-level read for every single byte.
            for raw_line in response.iter_lines(chunk_size=None, decode_unicode=True):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
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
                if max_reply_seconds > 0 and now - request_start > max_reply_seconds:
                    print(
                        f"[AI] llama.cpp reply exceeded {max_reply_seconds:.0f}s; stopping. "
                        "The GPU is probably out of VRAM (check nvidia-smi).",
                        flush=True,
                    )
                    cancelled = True
                    break

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
                        if self._cancel_event.is_set():
                            break
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
        if (
            not cancelled
            and remaining
            and on_sentence
            and self._is_speech_worthy(remaining)
        ):
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
                not cancelled
                and not full_text.strip()
                and output_tokens >= max_output_tokens
            ),
            "cancelled": cancelled,
        }

        self._messages.append({"role": "user", "content": text})
        self._messages.append(
            {"role": "assistant", "content": result["text"]}
        )
        self._trim_history()
        return result

    def generate_response(self, text, on_sentence=None, context=None):
        if not text:
            return ""

        request_start = time.perf_counter()
        # A new turn starts uncancelled; an interrupt that arrived while idle
        # must not cancel the next answer.
        self._cancel_event.clear()
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

        if attempt.get("cancelled"):
            print(
                f"[AI] Generation cancelled after {attempt['request_time']:.2f}s.",
                flush=True,
            )
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
