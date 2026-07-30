import sounddevice as sd


from . import config


class AudioPlayer:

    def __init__(self):

        self.sample_rate = config.SAMPLE_RATE

    def play(self, audio):

        print("\n▶ Playing recording...")

        sd.play(audio, self.sample_rate)

        sd.wait()

    def review(self):

        while True:

            choice = input(
                "\n[S] Save  [R] Retry  [K] Skip  [Q] Quit : "
            ).strip().lower()
 
            if choice in ("s", "r", "k", "q"):
                return choice

            print("Invalid choice.")

            while True:

                choice = input("> ").strip().lower()

                if choice in ("s", "r", "k", "q"):
                    return choice

                print("Invalid choice. Enter S, R, K or Q.")