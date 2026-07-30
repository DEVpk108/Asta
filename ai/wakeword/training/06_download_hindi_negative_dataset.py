"""
06_download_hindi_negative_dataset.py

Prepare Hindi speech negatives for ASTA.

Dataset:
    dhruvkys/11k-asr

The dataset is a soundfolder dataset containing:

    train/
        clips/
            *.mp3
        metadata.tsv

This script intentionally DOES NOT use:
    datasets.load_dataset()

That avoids Hugging Face's automatic Audio feature
decoding and therefore avoids the torchcodec / PyTorch
dependency.

Pipeline:

    Hugging Face snapshot
            ↓
    metadata.tsv
            ↓
    MP3 files
            ↓
    duration filtering
            ↓
    wakeword filtering
            ↓
    librosa decoding
            ↓
    mono
            ↓
    22050 Hz
            ↓
    16-bit PCM WAV
            ↓
    duplicate detection
            ↓
    manifest

Author: ASTA
"""

from pathlib import Path
import csv
import hashlib
import random
import re
import shutil
import wave

import numpy as np
from tqdm import tqdm


from ai.wakeword.config import (
    GENERATED_DIR,
)


# =========================================================
# Configuration
# =========================================================

DATASET_NAME = "dhruvkys/11k-asr"

DATASET_SPLIT = "train"

TARGET_SAMPLES = 10

TARGET_SAMPLE_RATE = 22050

MIN_DURATION_SECONDS = 0.5

MAX_DURATION_SECONDS = 5.0

RANDOM_SEED = 42


# =========================================================
# Paths
# =========================================================

OUTPUT_DIR = (
    GENERATED_DIR
    / "downloaded_negative"
    / "hindi"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "hindi_manifest.csv"
)

SOURCE_DIR = (
    GENERATED_DIR
    / "negative_sources"
    / "11k-asr"
)


# =========================================================
# Wakeword filtering
# =========================================================

WAKEWORD_PATTERNS = [

    # English
    r"\bhey\s+aasta\b",
    r"\bhey\s+asta\b",
    r"\bhello\s+aasta\b",
    r"\bhello\s+asta\b",
    r"\bhi\s+aasta\b",
    r"\bhi\s+asta\b",

    # Hindi / Devanagari
    r"हे\s+आस्ता",
    r"हे\s+आस्टा",
    r"हे\s+अस्ता",
    r"हाय\s+आस्ता",
    r"हेलो\s+आस्ता",
]


# =========================================================
# Text utilities
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    ).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def contains_wakeword(
    text: str,
) -> bool:

    normalized = normalize_text(
        text
    )

    for pattern in WAKEWORD_PATTERNS:

        if re.search(
            pattern,
            normalized,
        ):

            return True

    return False


# =========================================================
# Hugging Face download
# =========================================================

def download_dataset():

    try:

        from huggingface_hub import (
            snapshot_download,
        )

    except ImportError:

        raise RuntimeError(
            "\n"
            "huggingface_hub is required.\n\n"
            "Install with:\n"
            "pip install huggingface_hub\n"
        )

    SOURCE_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Downloading Hindi dataset files..."
    )

    print(
        f"Dataset : {DATASET_NAME}"
    )

    print(
        f"Destination : {SOURCE_DIR}"
    )

    print()

    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=str(
            SOURCE_DIR
        ),
        local_dir_use_symlinks=False,
    )

    return SOURCE_DIR


# =========================================================
# Locate metadata
# =========================================================

def find_metadata(
    root: Path,
) -> Path:

    candidates = [

        root
        / DATASET_SPLIT
        / "metadata.tsv",

        root
        / "metadata.tsv",

    ]

    for path in candidates:

        if path.exists():

            return path

    matches = list(
        root.rglob(
            "metadata.tsv"
        )
    )

    if not matches:

        raise FileNotFoundError(
            "Could not find metadata.tsv "
            f"inside:\n{root}"
        )

    return matches[0]


# =========================================================
# Locate clips
# =========================================================

def find_clips_dir(
    root: Path,
) -> Path:

    candidates = [

        root
        / DATASET_SPLIT
        / "clips",

        root
        / "clips",

    ]

    for path in candidates:

        if path.exists():

            return path

    matches = list(
        root.rglob(
            "clips"
        )
    )

    for path in matches:

        if path.is_dir():

            return path

    raise FileNotFoundError(
        "Could not find clips directory "
        f"inside:\n{root}"
    )


# =========================================================
# Read metadata
# =========================================================

def read_metadata(
    metadata_path: Path,
):

    rows = []

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if not reader.fieldnames:

            raise ValueError(
                "metadata.tsv has no header."
            )

        # The official dataset uses:
        #
        # file_name
        # text

        file_column = None

        text_column = None

        for name in reader.fieldnames:

            normalized = (
                name.strip().lower()
            )

            if normalized in (
                "file_name",
                "filename",
                "file",
            ):

                file_column = name

            elif normalized in (
                "text",
                "sentence",
                "transcription",
                "transcript",
            ):

                text_column = name

        if file_column is None:

            raise ValueError(
                "Could not find file_name "
                "column in metadata.tsv."
            )

        if text_column is None:

            raise ValueError(
                "Could not find text "
                "column in metadata.tsv."
            )

        for row in reader:

            filename = (
                row.get(
                    file_column,
                    "",
                )
                or ""
            ).strip()

            text = (
                row.get(
                    text_column,
                    "",
                )
                or ""
            ).strip()

            if not filename:

                continue

            rows.append(
                (
                    filename,
                    text,
                )
            )

    return rows


# =========================================================
# Audio loading
# =========================================================

def load_audio(
    path: Path,
):

    try:

        import librosa

    except ImportError:

        raise RuntimeError(
            "\n"
            "librosa is required.\n\n"
            "Install with:\n"
            "pip install librosa\n"
        )

    audio, sample_rate = (
        librosa.load(
            str(path),
            sr=None,
            mono=False,
        )
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    return (
        audio,
        int(sample_rate),
    )


# =========================================================
# Mono
# =========================================================

def to_mono(
    audio: np.ndarray,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 1:

        return audio

    if audio.ndim == 2:

        # librosa:
        # channels x samples

        if audio.shape[0] <= 8:

            return np.mean(
                audio,
                axis=0,
            )

        # Fallback:
        # samples x channels

        return np.mean(
            audio,
            axis=1,
        )

    raise ValueError(
        f"Unsupported audio shape: "
        f"{audio.shape}"
    )


# =========================================================
# Normalize
# =========================================================

def normalize_audio(
    audio: np.ndarray,
):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.clip(
        audio,
        -1.0,
        1.0,
    )


# =========================================================
# Resampling
# =========================================================

def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
):

    if source_rate == target_rate:

        return audio.astype(
            np.float32,
            copy=False,
        )

    try:

        import librosa

        output = librosa.resample(
            audio,
            orig_sr=source_rate,
            target_sr=target_rate,
        )

        return np.asarray(
            output,
            dtype=np.float32,
        )

    except Exception:

        # Safe interpolation fallback.

        if len(audio) == 0:

            return np.array(
                [],
                dtype=np.float32,
            )

        target_length = int(
            round(
                len(audio)
                * target_rate
                / source_rate
            )
        )

        old_positions = np.linspace(
            0.0,
            1.0,
            len(audio),
            endpoint=False,
        )

        new_positions = np.linspace(
            0.0,
            1.0,
            target_length,
            endpoint=False,
        )

        return np.interp(
            new_positions,
            old_positions,
            audio,
        ).astype(
            np.float32
        )


# =========================================================
# WAV writer
# =========================================================

def save_wav(
    path: Path,
    audio: np.ndarray,
):

    audio = normalize_audio(
        audio
    )

    pcm = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav:

        wav.setnchannels(1)

        wav.setsampwidth(2)

        wav.setframerate(
            TARGET_SAMPLE_RATE
        )

        wav.writeframes(
            pcm.tobytes()
        )


# =========================================================
# Hash
# =========================================================

def audio_hash(
    audio: np.ndarray,
):

    audio = normalize_audio(
        audio
    )

    pcm = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    return hashlib.sha256(
        pcm.tobytes()
    ).hexdigest()


# =========================================================
# Main
# =========================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    print()
    print("=" * 50)
    print(
        "ASTA HINDI NEGATIVE DATASET"
    )
    print("=" * 50)

    print(
        f"Dataset : {DATASET_NAME}"
    )

    print(
        f"Target  : {TARGET_SAMPLES}"
    )

    print(
        f"Sample rate : "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        f"Output  : {OUTPUT_DIR}"
    )

    print("=" * 50)

    # -----------------------------------------------------
    # Download source dataset
    # -----------------------------------------------------

    dataset_root = (
        download_dataset()
    )

    # -----------------------------------------------------
    # Locate files
    # -----------------------------------------------------

    metadata_path = (
        find_metadata(
            dataset_root
        )
    )

    clips_dir = (
        find_clips_dir(
            dataset_root
        )
    )

    print()
    print(
        f"Metadata : {metadata_path}"
    )

    print(
        f"Clips    : {clips_dir}"
    )

    # -----------------------------------------------------
    # Read metadata
    # -----------------------------------------------------

    rows = read_metadata(
        metadata_path
    )

    print(
        f"Usable metadata rows : "
        f"{len(rows)}"
    )

    # -----------------------------------------------------
    # Shuffle source order
    # -----------------------------------------------------

    random.shuffle(
        rows
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    generated = 0

    duration_filtered = 0

    duplicates = 0

    conversion_errors = 0

    text_filtered = 0

    missing_files = 0

    seen_hashes = set()

    manifest_rows = []

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    progress = tqdm(
        total=TARGET_SAMPLES,
        desc="Preparing Hindi negatives",
        unit="clip",
    )

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    for filename, text in rows:

        if generated >= TARGET_SAMPLES:

            break

        # -------------------------------------------------
        # Wakeword protection
        # -------------------------------------------------

        if contains_wakeword(
            text
        ):

            text_filtered += 1

            continue

        # -------------------------------------------------
        # Locate source audio
        # -------------------------------------------------

        source_path = (
            clips_dir
            / filename
        )

        if not source_path.exists():

            missing_files += 1

            continue

        # -------------------------------------------------
        # Load audio
        # -------------------------------------------------

        try:

            audio, source_rate = (
                load_audio(
                    source_path
                )
            )

            audio = to_mono(
                audio
            )

            audio = normalize_audio(
                audio
            )

        except Exception:

            conversion_errors += 1

            continue

        if source_rate <= 0:

            conversion_errors += 1

            continue

        # -------------------------------------------------
        # Duration
        # -------------------------------------------------

        source_duration = (
            len(audio)
            / source_rate
        )

        if (
            source_duration
            < MIN_DURATION_SECONDS
            or
            source_duration
            > MAX_DURATION_SECONDS
        ):

            duration_filtered += 1

            continue

        # -------------------------------------------------
        # Resample
        # -------------------------------------------------

        try:

            audio = resample_audio(
                audio,
                source_rate,
                TARGET_SAMPLE_RATE,
            )

            audio = normalize_audio(
                audio
            )

        except Exception:

            conversion_errors += 1

            continue

        # -------------------------------------------------
        # Duplicate detection
        # -------------------------------------------------

        digest = audio_hash(
            audio
        )

        if digest in seen_hashes:

            duplicates += 1

            continue

        seen_hashes.add(
            digest
        )

        # -------------------------------------------------
        # Output filename
        # -------------------------------------------------

        output_filename = (
            f"hindi_"
            f"{generated:06d}.wav"
        )

        output_path = (
            OUTPUT_DIR
            / output_filename
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        try:

            save_wav(
                output_path,
                audio,
            )

        except Exception:

            conversion_errors += 1

            continue

        # -------------------------------------------------
        # Manifest
        # -----------------------------------------------------

        manifest_rows.append([
            output_filename,
            filename,
            text,
            f"{source_duration:.6f}",
            source_rate,
            TARGET_SAMPLE_RATE,
            digest,
        ])

        generated += 1

        progress.update(
            1
        )

    progress.close()

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "filename",
            "source_filename",
            "text",
            "source_duration_seconds",
            "source_sample_rate",
            "output_sample_rate",
            "sha256",
        ])

        writer.writerows(
            manifest_rows
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print(
        "HINDI NEGATIVE DATASET COMPLETE"
    )
    print("=" * 50)

    print(
        f"Generated          : "
        f"{generated}"
    )

    print(
        f"Target             : "
        f"{TARGET_SAMPLES}"
    )

    print(
        f"Text filtered      : "
        f"{text_filtered}"
    )

    print(
        f"Duration filtered  : "
        f"{duration_filtered}"
    )

    print(
        f"Missing files      : "
        f"{missing_files}"
    )

    print(
        f"Duplicates         : "
        f"{duplicates}"
    )

    print(
        f"Conversion errors  : "
        f"{conversion_errors}"
    )

    print(
        f"Manifest           : "
        f"{MANIFEST_PATH}"
    )

    print("=" * 50)

    if generated < TARGET_SAMPLES:

        print()
        print(
            "RESULT: INCOMPLETE"
        )

        print(
            "Could not reach the requested "
            "number of Hindi negatives."
        )

    elif conversion_errors:

        print()
        print(
            "RESULT: PASS WITH WARNINGS"
        )

    else:

        print()
        print(
            "RESULT: PASS"
        )

    print("=" * 50)


if __name__ == "__main__":

    main()