import random


class PromptManager:

    def __init__(self, repeats=200):

        base_prompts = [
            "Hey ASTA",
            "Hello ASTA",
            "Wake up ASTA",
        ]


        self.prompts = []

        for prompt in base_prompts:
            self.prompts.extend([prompt] * repeats)

        random.shuffle(self.prompts)

        self.current = 0
        self.total = len(self.prompts)

    def has_next(self):
        return self.current < self.total

    def next(self):

        if not self.has_next():
            raise StopIteration("No prompts remaining.")

        prompt = self.prompts[self.current]
        self.current += 1

        return prompt

    def progress(self):
        return self.current, self.total

    def remaining(self):
        return self.total - self.current

    def reset(self):

        self.current = 0
        random.shuffle(self.prompts)