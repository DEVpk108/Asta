from colorama import Fore
from colorama import Style
from colorama import init

from tqdm import tqdm

init(autoreset=True)


class DatasetUI:

    def __init__(self, total):

        self.bar = tqdm(
            total=total,
            dynamic_ncols=True,
            colour="green",
        )

    # --------------------------------------------------

    def banner(self):

        print("=" * 60)
        print("ASTA Wakeword Dataset Creator")
        print("=" * 60)

    # --------------------------------------------------

    def show_prompt(self, prompt):

        print()
        print(Fore.CYAN + f'Prompt : "{prompt}"')

    # --------------------------------------------------

    def update(self):

        self.bar.update(1)

    # --------------------------------------------------

    def stats(self, session):

        print()

        print(
            Fore.GREEN
            + f"Saved : {session.saved}"
        )

        print(
            Fore.YELLOW
            + f"Skipped : {session.skipped}"
        )

        print(
            Fore.CYAN
            + f"Remaining : {session.remaining}"
        )

    # --------------------------------------------------

    def saved(self, filename):

        print(
            Fore.GREEN
            + f"✔ Saved -> {filename}"
        )

    def skipped(self):

        print(
            Fore.YELLOW
            + "Skipped."
        )

    def retry(self):

        print(
            Fore.RED
            + "Retry recording..."
        )

    def recording(self):

        print(
            Fore.CYAN
            + "Recording..."
        )

    def playback(self):

        print(
            Fore.MAGENTA
            + "Playing recording..."
        )

    # --------------------------------------------------

    def close(self):

        self.bar.close()
        
    def speech_detected(self):
        print(Fore.GREEN + "Speech detected.")
        
    def speech_timeout(self):
        print(Fore.RED + "No speech detected.")
        
        
    def recording_finished(self):
        print(Fore.GREEN + "Recording complete.")