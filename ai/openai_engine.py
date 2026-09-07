import json
import time

import requests


class AIEngine:

    def __init__(
        self,
        base_url="http://localhost:1234",
        model="nvidia/nemotron-3-nano-4b",
        timeout=120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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

    def generate_response(self, text, on_sentence=None):
        if not text:
            return ""

        payload = {
            "model": self.model,
            "input": text,
            "stream": True,
            "store": True,
            # Nemotron is reasoning-heavy, so leave room for both reasoning
            # and the visible assistant message.
            "max_output_tokens": 256,
        }

        if self.previous_response_id is None:
            payload["system_prompt"] = self.system_prompt
        else:
            payload["previous_response_id"] = self.previous_response_id

        request_start = time.perf_counter()
        first_token_time = None
        full_text = ""
        sentence_buffer = ""
        final_result = None

        try:
            with requests.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
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
                        delta = data.get("content", "")
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
                                on_sentence(sentence)

                    elif event_name == "chat.end":
                        final_result = data.get("result", {})

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

        if final_result:
            response_id = final_result.get("response_id")
            if response_id:
                self.previous_response_id = response_id
                print("[AI] Conversation state updated.", flush=True)

            # Some reasoning models may spend the streaming budget on
            # reasoning and emit the visible message only in chat.end.
            if not full_text.strip():
                full_text = self._extract_message_text(final_result.get("output"))
                if full_text and on_sentence:
                    # Emit the final message as one sentence/chunk when no
                    # message.delta events were received.
                    on_sentence(full_text)

        remaining = sentence_buffer.strip()
        if remaining:
            print(f"[AI] Sentence: {remaining}", flush=True)
            if on_sentence:
                on_sentence(remaining)

        stats = final_result.get("stats", {}) if final_result else {}
        input_tokens = stats.get("input_tokens", 0)
        output_tokens = stats.get("total_output_tokens", 0)
        reasoning_tokens = stats.get("reasoning_output_tokens", 0)
        tokens_per_second = stats.get("tokens_per_second", 0.0)
        model_load_time = stats.get("model_load_time_seconds")

        total_request_time = time.perf_counter() - request_start
        print(f"[AI] Request: {total_request_time:.2f}s", flush=True)
        print(f"[AI] Input tokens: {input_tokens}", flush=True)
        print(f"[AI] Output tokens: {output_tokens}", flush=True)
        print(f"[AI] Reasoning tokens: {reasoning_tokens}", flush=True)
        print(f"[AI] LM Studio speed: {tokens_per_second:.2f} tok/s", flush=True)

        if model_load_time is not None:
            print(f"[AI] Model load: {model_load_time:.3f}s", flush=True)

        print(f"[AI] Response: {full_text.strip()}", flush=True)
        return full_text.strip()
