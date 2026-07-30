from pathlib import Path
import time
import openwakeword

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "wakeword_dataset_v1"

POSITIVE_DIR = DATASET_ROOT / "positive"
NEGATIVE_DIR = DATASET_ROOT / "negative_wav"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_NAME = "asta"

# ------------------------------------------------------------------


def collect_audio(folder: Path):

    return sorted(str(f) for f in folder.rglob("*.wav"))


def print_summary(positive, negative):

    print("\n==============================")
    print("ASTA TRAINING SUMMARY")
    print("==============================")

    print(f"Positive clips : {len(positive)}")
    print(f"Negative clips : {len(negative)}")

    print(f"\nPositive Folder : {POSITIVE_DIR}")
    print(f"Negative Folder : {NEGATIVE_DIR}")

    print(f"\nOutput Folder   : {MODEL_DIR}")
    print(f"Model Name      : {MODEL_NAME}")

    print("==============================\n")


def main():

    MODEL_DIR.mkdir(exist_ok=True)

    positive = collect_audio(POSITIVE_DIR)
    negative = collect_audio(NEGATIVE_DIR)

    if not positive:
        raise RuntimeError("No positive WAV files found.")

    if not negative:
        raise RuntimeError("No negative WAV files found.")

    print_summary(positive, negative)

    start = time.time()

    print("Training verifier...\n")

    openwakeword.train_custom_verifier(

        positive_reference_clips=positive,

        negative_reference_clips=negative,

        output_path=str(MODEL_DIR),

        model_name=MODEL_NAME

    )

    elapsed = time.time() - start

    print("\nTraining completed!")

    print(f"Elapsed Time : {elapsed:.2f} seconds")

    print(f"\nVerifier saved to:\n{MODEL_DIR}")


if __name__ == "__main__":
    main()