from voice.microphone_engine import MicrophoneEngine
from voice.vad_engine import VADEngine

from .recorder import DatasetRecorder
from .prompts import PromptManager
from .metadata import MetadataManager
from .player import AudioPlayer
from .session import RecordingSession


def main():

    print("=" * 60)
    print("ASTA Wakeword Dataset Creator")
    print("=" * 60)

    # ------------------------
    # Initialize
    # ------------------------

    mic = MicrophoneEngine()
    vad = VADEngine()

    mic.start()

    recorder = DatasetRecorder(mic, vad)
    prompts = PromptManager()
    session = RecordingSession(
        prompts.prompts
    )
    metadata = MetadataManager()
    player = AudioPlayer()
    
    
    if session.load():

        print()

        print("Previous session found.")

        choice = input("Resume? (Y/n): ").strip().lower()

        if choice == "n":

            session.finish()

            session = RecordingSession(
                prompts.prompts
            )     
    
    
    

    try:

        while session.has_next():

            current = session.current
            total = session.total
            prompt = session.current_prompt()

            print()
            print("-" * 60)
            print(f"Sample {current + 1}/{total}")
            print(f'Prompt : "{prompt}"')
            print("-" * 60)

            audio = recorder.record(prompt)

            if audio is None:
                print("Recording failed.")
                continue

            while True:

                try:
                    player.play(audio)
                except Exception as e:
                    print(f"[Player] {e}")

                choice = player.review().lower()

                # ------------------------
                # Retry
                # ------------------------

                if choice == "r":

                    print("Retrying...\n")

                    new_audio = recorder.record(prompt)

                    if new_audio is None:
                        print("Recording failed. Try again.")
                        continue

                    audio = new_audio
                    continue

                # ------------------------
                # Save
                # ------------------------

                elif choice == "s":

                    filename = metadata.save(audio, prompt)
                    session.mark_saved()

                    print(f"✔ Saved -> {filename}")

                    break

                # ------------------------
                # Skip
                # ------------------------

                elif choice == "k":
                    session.mark_saved()

                    print("Skipped.")

                    break

                # ------------------------
                # Quit
                # ------------------------

                elif choice == "q":

                    print("\nExiting Dataset Creator...")

                    return

                # ------------------------
                # Invalid Input
                # ------------------------

                else:

                    print("Invalid choice.")
                    print("Choose: [S] Save  [R] Retry  [K] Skip  [Q] Quit")
                
                if not session.has_next():

                    session.finish()

                    print()
                    print("Dataset completed!")
                    print(f"Saved   : {session.saved}")
                    print(f"Skipped : {session.skipped}")

    except KeyboardInterrupt:

        print("\nStopping Dataset Creator...")
        
    finally:

        mic.stop()

        print("Microphone stopped.")


if __name__ == "__main__":
    main()