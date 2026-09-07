import json
import time

import requests


class AIEngine:

    def __init__(
        self,
        base_url="http://localhost:1234",
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

        self.chat_url = f"{self.base_url}/api/v1/chat"
        self.load_url = f"{self.base_url}/api/v1/models/load"
        self.previous_response_id = None
        self.warmed = False

        self.system_prompt = (
            "You are ASTA, a local AI voice assistant. "
            "Respond naturally and concisely. "
            "Prefer 1–3 short sentences for normal questions. "
            "Avoid long explanations unless the user asks for detail. "
            "Sound conversational, helpful, and direct."
        )

    def warmup(self):
        """Load the selected model and run a tiny throwaway generation.

        The explicit model load removes first-request model-loading work from a
        user's interaction. The one-token chat then warms the inference path
        without creating/storing conversation state.
        """
        start = time.perf_counter()
        try:
            load_start = time.perf_counter()
            response = self.session.post(
                self.load_url,
                json={"model": self.model},
                timeout=self.timeout,
            )
            response.raise_for_status()
            print(
                f"[AI] Model load request: {time.perf_counter() - load_start:.3f}s",
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

                if event_name == "chat.start":
                    chat_start_time = now
                    continue
                if event_name == "model_load.start":
                    model_load_start = now
                    continue
                if event_name == "model_load.end":
                    model_load_end = now
                    continue
                if event_name == "prompt_processing.start":
                    prompt_start = now
                    continue
                if event_name == "prompt_processing.end":
                    prompt_end = now
                    continue
                if event_name == "message.start":
                    message_start = now
                    continue
                if event_name == "chat.end":
                    final_result = data.get("result", {})
                    continue
                if event_name != "message.delta":
                    continue

                delta = self._normalize_text(data.get("content", ""))
                if not delta:
                    continue

                if first_delta_time is None:
                    first_delta_time = now
                    print(
                        f"[AI] Client TTFT: {first_delta_time - request_start:.3f}s",
                        flush=True,
                    )

                full_text += delta
                sentence_buffer += delta

                while True:
                    sentence_end = None
                    for punctuation in (".", "!", "?"):
                        index = sentence_buffer.find(punctuation)
                        if index != -1 and (sentence_end is None or index < sentence_end):
                            sentence_end = index
                    if sentence_end is None:
                        break

                    sentence = sentence_buffer[:sentence_end + 1].strip()
                    sentence_buffer = sentence_buffer[sentence_end + 1:]
                    sentence = self._normalize_text(sentence)
                    if on_sentence and self._is_speech_worthy(sentence):
                        on_sentence(sentence)

        if final_result:
            message_text = self._normalize_text(self._extract_message_text(final_result.get("output")))
            if not full_text.strip() and message_text:
                full_text = message_text
                if on_sentence and self._is_speech_worthy(full_text):
                    on_sentence(full_text)

        remaining = self._normalize_text(sentence_buffer.strip())
        if remaining and on_sentence and self._is_speech_worthy(remaining):
            on_sentence(remaining)

        input_tokens, output_tokens, reasoning_tokens, tokens_per_second, ttft, model_load_time = self._output_stats(final_result)
        total_request_time = time.perf_counter() - request_start

        return {
            "text": self._normalize_text(full_text.strip()),
            "result": final_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tokens_per_second": tokens_per_second,
            "ttft": ttft,
            "model_load_time": model_load_time,
            "request_time": total_request_time,
            "response_open_time": (
                response_open - request_start if response_open is not None else None
            ),
            "first_event_time": (
                first_event_time - request_start if first_event_time is not None else None
            ),
            "chat_start_time": (
                chat_start_time - request_start if chat_start_time is not None else None
            ),
            "model_load_event_time": (
                model_load_end - model_load_start
                if model_load_start is not None and model_load_end is not None
                else None
            ),
            "prompt_processing_time": (
                prompt_end - prompt_start
                if prompt_start is not None and prompt_end is not None
                else None
            ),
            "prompt_end_to_message_start": (
                message_start - prompt_end
                if prompt_end is not None and message_start is not None
                else None
            ),
            "message_start_to_first_delta": (
                first_delta_time - message_start
                if first_delta_time is not None and message_start is not None
                else None
            ),
            "request_to_first_delta": (
                first_delta_time - request_start if first_delta_time is not None else None
            ),
            "exhausted_reasoning": (
                not full_text.strip()
                and output_tokens >= max_output_tokens
                and reasoning_tokens >= max_output_tokens
            ),
        }

    def generate_response(self, text, on_sentence=None):
        if not text:
            return ""

        request_start = time.perf_counter()
        try:
            attempt = self._request(text, on_sentence, self.max_output_tokens)
            if attempt["exhausted_reasoning"] and self.reasoning_retry_tokens > self.max_output_tokens:
                print(
                    f"[AI] Output budget exhausted; retrying with {self.reasoning_retry_tokens} output tokens.",
                    flush=True,
                )
                attempt = self._request(text, on_sentence, self.reasoning_retry_tokens)
        except requests.RequestException as exc:
            print(
                f"[AI] Connection error after {time.perf_counter() - request_start:.2f}s: {exc}",
                flush=True,
            )
            return ""
        except Exception as exc:
            print(
                f"[AI] Error after {time.perf_counter() - request_start:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""

        final_result = attempt["result"]
        if final_result and final_result.get("response_id"):
            self.previous_response_id = final_result["response_id"]
            print("[AI] Conversation state updated.", flush=True)

        print(f"[AI] Request: {attempt['request_time']:.2f}s", flush=True)
        print(f"[AI] Input tokens: {attempt['input_tokens']}", flush=True)
        print(f"[AI] Output tokens: {attempt['output_tokens']}", flush=True)
        print(f"[AI] Reasoning tokens: {attempt['reasoning_tokens']}", flush=True)
        print(f"[AI] LM Studio speed: {attempt['tokens_per_second']:.2f} tok/s", flush=True)
        if attempt["ttft"] is not None:
            print(f"[AI] LM Studio reported TTFT: {attempt['ttft']:.3f}s", flush=True)
        if attempt["response_open_time"] is not None:
            print(f"[AI] HTTP headers received: {attempt['response_open_time']:.3f}s", flush=True)
        if attempt["first_event_time"] is not None and attempt["response_open_time"] is not None:
            print(
                f"[AI] Headers -> first SSE event: "
                f"{attempt['first_event_time'] - attempt['response_open_time']:.3f}s",
                flush=True,
            )
        if attempt["chat_start_time"] is not None and attempt["response_open_time"] is not None:
            print(
                f"[AI] Headers -> chat.start: "
                f"{attempt['chat_start_time'] - attempt['response_open_time']:.3f}s",
                flush=True,
            )
        if attempt["model_load_event_time"] is not None:
            print(f"[AI] Model load event: {attempt['model_load_event_time']:.3f}s", flush=True)
        if attempt["prompt_processing_time"] is not None:
            print(f"[AI] Prompt processing: {attempt['prompt_processing_time']:.3f}s", flush=True)
        if attempt["prompt_end_to_message_start"] is not None:
            print(
                f"[AI] prompt.end -> message.start: {attempt['prompt_end_to_message_start']:.3f}s",
                flush=True,
            )
        if attempt["message_start_to_first_delta"] is not None:
            print(
                f"[AI] Message start -> first delta: {attempt['message_start_to_first_delta']:.3f}s",
                flush=True,
            )
        if attempt["model_load_time"] is not None:
            print(f"[AI] Model load: {attempt['model_load_time']:.3f}s", flush=True)
        print(f"[AI] Response: {attempt['text']}", flush=True)
        return attempt["text"]
