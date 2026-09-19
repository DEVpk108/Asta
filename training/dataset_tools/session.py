from pathlib import Path
import json


class RecordingSession:

    def __init__(self, prompts):

        self.prompts = prompts

        self.total = len(prompts)

        self.current = 0
        self.saved = 0
        self.skipped = 0

        self.session_file = Path("wakeword_dataset/session.json")
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------

    def has_next(self):
        return self.current < self.total

    def current_prompt(self):

        if not self.has_next():
            return None

        return self.prompts[self.current]

    def advance(self):
        self.current += 1

    # ------------------------

    @property
    def remaining(self):
        return self.total - self.current

    # ------------------------

    def mark_saved(self):

        self.saved += 1
        self.advance()
        self.save()

    def mark_skipped(self):

        self.skipped += 1
        self.advance()
        self.save()

    # ------------------------

    def save(self):

        data = {
            "current": self.current,
            "saved": self.saved,
            "skipped": self.skipped,
            "prompts": self.prompts,
        }

        with open(self.session_file, "w") as f:
            json.dump(data, f, indent=4)

    
    def load(self):

        if not self.session_file.exists():
            return False

        try:

            with open(self.session_file, "r") as f:
                data = json.load(f)

            self.current = data.get("current", 0)
            self.saved = data.get("saved", 0)
            self.skipped = data.get("skipped", 0)

            if "prompts" in data:
                self.prompts = data["prompts"]
                self.total = len(self.prompts)

            return True

        except (json.JSONDecodeError, OSError):

            print("[Session] Invalid session. Starting new.")

            self.reset()

            return False

    # ------------------------

    def finish(self):

        if self.session_file.exists():
            self.session_file.unlink()