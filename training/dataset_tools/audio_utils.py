import sounddevice as sd


class AudioPlayer:

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def beep(self):
        print("\a", end="", flush=True)

    def play(self, audio):

        print("\nPlaying recording...")

        sd.play(audio, self.sample_rate)
        sd.wait()

    def review(self):

        while True:

            print("\nChoose an option:")

            print("[S] Save")
            print("[R] Retry")
            print("[K] Skip")
            print("[Q] Quit")

            choice = input("> ").strip().lower()

            if choice in ("s", "r", "k", "q"):
                return choice

            print("Invalid choice.")