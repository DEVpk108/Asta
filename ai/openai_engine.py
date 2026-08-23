import json
import time

import requests


class AIEngine:

    def __init__(
        self,
        base_url="http://localhost:1234",
        model="qwen/qwen3.5-9b",
        timeout=120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        self.chat_url = (
            f"{self.base_url}/api/v1/chat"
        )

        # LM Studio conversation state.
        self.previous_response_id = None

        self.system_prompt = (
            "You are ASTA, a local AI voice assistant. "
            "Respond naturally and concisely. "
            "Prefer 1–3 short sentences for normal questions. "
            "Avoid long explanations unless the user asks "
            "for detail. "
            "Sound conversational, helpful, and direct."
        )

    # ---------------------------------------------------------
    # Conversation control
    # ---------------------------------------------------------

    def reset_conversation(self):
        """Start a fresh conversation."""
        self.previous_response_id = None

        print(
            "[AI] Conversation memory reset.",
            flush=True,
        )

    # ---------------------------------------------------------
    # Generate response
    # ---------------------------------------------------------

    def generate_response(
        self,
        text,
        on_sentence=None,
    ):
        if not text:
            return ""

        payload = {
            "model": self.model,
            "input": text,
            "stream": True,
            "store": True,
            "max_output_tokens": 80,
        }

        # -----------------------------------------------------
        # Start a new conversation with the system prompt.
        # -----------------------------------------------------

        if self.previous_response_id is None:
            payload["system_prompt"] = (
                self.system_prompt
            )

        # -----------------------------------------------------
        # Continue the existing conversation.
        # -----------------------------------------------------

        else:
            payload["previous_response_id"] = (
                self.previous_response_id
            )

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

                for raw_line in response.iter_lines(
                    decode_unicode=True
                ):
                    if not raw_line:
                        continue

                    # -------------------------------------------------
                    # SSE event name
                    # -------------------------------------------------

                    if raw_line.startswith("event:"):
                        event_type = (
                            raw_line[
                                len("event:"):
                            ].strip()
                        )
                        continue

                    # -------------------------------------------------
                    # SSE JSON payload
                    # -------------------------------------------------

                    if not raw_line.startswith("data:"):
                        continue

                    data_text = (
                        raw_line[
                            len("data:"):
                        ].strip()
                    )

                    try:
                        data = json.loads(
                            data_text
                        )
                    except json.JSONDecodeError:
                        continue

                    event_name = (
                        data.get("type")
                        or event_type
                    )

                    # -------------------------------------------------
                    # Message token
                    # -------------------------------------------------

                    if event_name == "message.delta":

                        delta = data.get(
                            "content",
                            "",
                        )

                        if not delta:
                            continue

                        if first_token_time is None:
                            first_token_time = (
                                time.perf_counter()
                            )

                            ttft = (
                                first_token_time
                                - request_start
                            )

                            print(
                                f"[AI] TTFT: "
                                f"{ttft:.3f}s",
                                flush=True,
                            )

                        full_text += delta
                        sentence_buffer += delta

                        # -------------------------------------------------
                        # Detect complete sentences
                        # -------------------------------------------------

                        while True:

                            sentence_end = None

                            for punctuation in (
                                ".",
                                "!",
                                "?",
                            ):
                                index = (
                                    sentence_buffer.find(
                                        punctuation
                                    )
                                )

                                if (
                                    index != -1
                                    and (
                                        sentence_end
                                        is None
                                        or index
                                        < sentence_end
                                    )
                                ):
                                    sentence_end = (
                                        index
                                    )

                            if sentence_end is None:
                                break

                            sentence = (
                                sentence_buffer[
                                    :sentence_end + 1
                                ]
                                .strip()
                            )

                            sentence_buffer = (
                                sentence_buffer[
                                    sentence_end + 1:
                                ]
                            )

                            if sentence:
                                print(
                                    f"[AI] Sentence: "
                                    f"{sentence}",
                                    flush=True,
                                )

                                if on_sentence:
                                    on_sentence(
                                        sentence
                                    )

                    # -------------------------------------------------
                    # Final aggregated response
                    # -------------------------------------------------

                    elif event_name == "chat.end":

                        final_result = (
                            data.get(
                                "result",
                                {},
                            )
                        )

        except requests.RequestException as exc:

            elapsed = (
                time.perf_counter()
                - request_start
            )

            print(
                f"[AI] Connection error after "
                f"{elapsed:.2f}s: {exc}",
                flush=True,
            )

            return ""

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - request_start
            )

            print(
                f"[AI] Error after "
                f"{elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            return ""

        # ---------------------------------------------------------
        # Save LM Studio conversation state
        # ---------------------------------------------------------

        if final_result:
            response_id = final_result.get(
                "response_id"
            )

            if response_id:
                self.previous_response_id = (
                    response_id
                )

                print(
                    "[AI] Conversation state updated.",
                    flush=True,
                )

        # ---------------------------------------------------------
        # Flush unfinished final sentence
        # ---------------------------------------------------------

        remaining = sentence_buffer.strip()

        if remaining:

            print(
                f"[AI] Sentence: {remaining}",
                flush=True,
            )

            if on_sentence:
                on_sentence(remaining)

        # ---------------------------------------------------------
        # LM Studio telemetry
        # ---------------------------------------------------------

        stats = (
            final_result.get(
                "stats",
                {},
            )
            if final_result
            else {}
        )

        input_tokens = stats.get(
            "input_tokens",
            0,
        )

        output_tokens = stats.get(
            "total_output_tokens",
            0,
        )

        reasoning_tokens = stats.get(
            "reasoning_output_tokens",
            0,
        )

        tokens_per_second = stats.get(
            "tokens_per_second",
            0.0,
        )

        model_load_time = stats.get(
            "model_load_time_seconds"
        )

        total_request_time = (
            time.perf_counter()
            - request_start
        )

        print(
            f"[AI] Request: "
            f"{total_request_time:.2f}s",
            flush=True,
        )

        print(
            f"[AI] Input tokens: "
            f"{input_tokens}",
            flush=True,
        )

        print(
            f"[AI] Output tokens: "
            f"{output_tokens}",
            flush=True,
        )

        print(
            f"[AI] Reasoning tokens: "
            f"{reasoning_tokens}",
            flush=True,
        )

        print(
            f"[AI] LM Studio speed: "
            f"{tokens_per_second:.2f} tok/s",
            flush=True,
        )

        if model_load_time is not None:
            print(
                f"[AI] Model load: "
                f"{model_load_time:.3f}s",
                flush=True,
            )

        print(
            f"[AI] Response: "
            f"{full_text.strip()}",
            flush=True,
        )

        return full_text.strip()