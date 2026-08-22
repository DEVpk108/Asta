from openai import OpenAI


class AIEngine:

    def __init__(
        self,
        base_url="http://localhost:1234/v1",
        model="qwen/qwen3.5-9b",
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",
        )

        self.model = model

    def generate_response(self, text):

        if not text:
            return ""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are ASTA, an intelligent local AI "
                            "assistant. Respond naturally, clearly, "
                            "and concisely."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(
                f"[AI] {type(e).__name__}: {e}"
            )
            return ""