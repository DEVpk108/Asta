import json
import time

import requests


class AIEngine:

    def __init__(
        self,
        base_url="http://localhost:1234",
        model="nvidia/nemotron-3-nano-4b",
        timeout=120,
        max_output_tokens=512,
        reasoning_retry_tokens=1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.reasoning_retry_tokens = reasoning_retry_tokens

        self.chat_url = f"{self.base_url}/api/v1/chat"

        # LM Studio conversation state.
        self.previous_response_id = None

        self.system_prompt = (
            "You are ASTA, a local AI voice assistant. "
            "Respond naturally and concisely. "
            "Prefer 1–3 short sentences for normal questions. "
            "Avoid long explanations unless the user asks for detail. "
            "Sound conversational, helpful, and direct."
        )

    def reset_conversation(self):
        """Start a fresh conversation."""
        self.previous_response_id = None
        print("[AI] Conversation memory reset.", flush=True)

    @staticmethod
    def _extract_message_text(output):
        """Extract visible assistant text from LM Studio v1 output items."""
        if isinstance(output, str):
            return output.strip()

        if not isinstance(output, list):
            return ""

        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
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
            stats.get("model_load_time_seconds"),
        )

    @staticmethod
    def _normalize_text(text):
        """Repair common UTF-8-as-Windows-1252 mojibake without touching valid Unicode."""
        if not isinstance(text, str) or "\u00c3" not in text and "\u00e2" not in text:
            return text

        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

        return repaired if repaired != text else text

    def _request(self, text, on_sentence, max_output_tokens):
        payload = {
            "model": self.model,
            "input": text,
            "stream": True,
            "store": True,
            "max_output_tokens": max_output_tokens,
        }

        conversation_id = self.previous_response_id
        if conversation_id is None:
            payload["system_prompt"] = self.system_prompt
        else:
            payload["previous_response_id"] = conversation_id

        request_start = time.perf_counter()
        first_token_time = None
        full_text = ""
        sentence_buffer = ""
        final_result = None

        with requests.post(
            self.chat_url,
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()

            # LM Studio returns UTF-8 JSON/SSE. Requests can otherwise infer a
            # legacy single-byte encoding when the response omits charset.
            response.encoding = "utf-8"

            event_type = None

            for raw_line in response.iter_lines(decode_unicode=True):
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

                event_name = data.get("type") or event_type

                if event_name == "message.delta":
                    delta = self._normalize_text(data.get("content", ""))
                    if not delta:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft = first_token_time - request_start
                        print(f"[AI] TTFT: {ttft:.3f}s", flush=True)

                    full_text += delta
                    sentence_buffer += delta

                    while True:
                        sentence_end = None
                        for punctuation in (".", "!", "?"):
                            index = sentence_buffer.find(punctuation)
                            if index != -1 and (
                                sentence_end is None or index < sentence_end
                            ):
                                sentence_end = index

                        if sentence_end is None:
                            break

                        sentence = sentence_buffer[:sentence_end + 1].strip()
                        sentence_buffer = sentence_buffer[sentence_end + 1:]

                        if on_sentence and sentence:
                            on_sentence(self._normalize_text(sentence))

                elif event_name == "chat.end":
                    final_result = data.get("result", {})

        if final_result:
            message_text = self._normalize_text(
                self._extract_message_text(final_result.get("output"))
            )
            if not full_text.strip() and message_text:
                full_text = message_text
                if on_sentence:
                    on_sentence(full_text)

        remaining = self._normalize_text(sentence_buffer.strip())
        if remaining:
            print(f"[AI] Sentence: {remaining}", flush=True)
            if on_sentence:
                on_sentence(remaining)

        input_tokens, output_tokens, reasoning_tokens, tokens_per_second, model_load_time = self._output_stats(final_result)
        total_request_time = time.perf_counter() - request_start

        return {
            "text": self._normalize_text(full_text.strip()),
            "result": final_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tokens_per_second": tokens_per_second,
            "model_load_time": model_load_time,
            "request_time": total_request_time,
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
                    f"[AI] Reasoning budget exhausted; retrying with {self.reasoning_retry_tokens} output tokens.",
                    flush=True,
                )
                attempt = self._request(text, on_sentence, self.reasoning_retry_tokens)

        except requests.RequestException as exc:
            elapsed = time.perf_counter() - request_start
            print(f"[AI] Connection error after {elapsed:.2f}s: {exc}", flush=True)
            return ""
        except Exception as exc:
            elapsed = time.perf_counter() - request_start
            print(
                f"[AI] Error after {elapsed:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return ""

        final_result = attempt["result"]
        if final_result:
            response_id = final_result.get("response_id")
            if response_id:
                self.previous_response_id = response_id
                print("[AI] Conversation state updated.", flush=True)

        print(f"[AI] Request: {attempt['request_time']:.2f}s", flush=True)
        print(f"[AI] Input tokens: {attempt['input_tokens']}", flush=True)
        print(f"[AI] Output tokens: {attempt['output_tokens']}", flush=True)
        print(f"[AI] Reasoning tokens: {attempt['reasoning_tokens']}", flush=True)
        print(f"[AI] LM Studio speed: {attempt['tokens_per_second']:.2f} tok/s", flush=True)

        if attempt["model_load_time"] is not None:
            print(f"[AI] Model load: {attempt['model_load_time']:.3f}s", flush=True)

        print(f"[AI] Response: {attempt['text']}", flush=True)
        return attempt["text"]
