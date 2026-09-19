from pathlib import Path

# ================================
# Audio
# ================================

SAMPLE_RATE = 16000

# ================================
# Dataset
# ================================

DATASET_DIR = Path("wakeword_dataset")

POSITIVE_DIR = DATASET_DIR / "positive"
NEGATIVE_DIR = DATASET_DIR / "negative"

# Wakeword folders
POSITIVE_FOLDERS = {
    "Hey ASTA": POSITIVE_DIR / "hey_asta",
    "Hello ASTA": POSITIVE_DIR / "hello_asta",
    "Wake up ASTA": POSITIVE_DIR / "wake_up_asta",
}

NEGATIVE_RANDOM = NEGATIVE_DIR / "random"

METADATA_FILE = DATASET_DIR / "metadata.csv"

# Create folders automatically
for folder in POSITIVE_FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

NEGATIVE_RANDOM.mkdir(parents=True, exist_ok=True)