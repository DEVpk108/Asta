"""
05_download_negative_dataset.py

Download and prepare LibriSpeech dev-clean as negative speech
for ASTA wakeword training.

Source:
    LibriSpeech dev-clean

The archive is downloaded once and then processed locally.

Output:
    ai/wakeword/generated/downloaded_negative/

Author: ASTA
"""

from pathlib import Path
import csv
import hashlib
import random
import re
import shutil
import tarfile
import urllib.request
import wave

import numpy as np
from tqdm import tqdm

from ai.wakeword.config import GENERATED_DIR


# =========================================================
# Configuration
# =========================================================

LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/"
    "dev-clean.tar.gz"
)

ARCHIVE_NAME = "dev-clean.tar.gz"

ARCHIVE_PATH = (
    GENERATED_DIR
    / "negative_sources"
    / ARCHIVE_NAME
)

EXTRACT_DIR = (
    GENERATED_DIR
    / "negative_sources"
    / "LibriSpeech"
)

OUTPUT_DIR = (
    GENERATED_DIR
    / "downloaded_negative"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "librispeech_manifest.csv"
)

TARGET_SAMPLES = 10

MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 5.0

TARGET_SAMPLE_RATE = 22050

RANDOM_SEED = 42

# ---------------------------------------------------------
# Wakeword filtering
# ---------------------------------------------------------

WAKEWORD_PATTERNS = [

    r"\bhey\s+aasta\b",
    r"\bhey\s+asta\b",

    r"\bhello\s+aasta\b",
    r"\bhello\s+asta\b",

    r"\bhi\s+aasta\b",
    r"\bhi\s+asta\b",

]


# =========================================================
# Download helper
# =========================================================

def download_file(
    url: str,
    destination: Path,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():

        print()
        print(
            f"Archive already exists:"
        )
        print(
            destination
        )

        return

    print()
    print(
        "Downloading LibriSpeech "
        "dev-clean..."
    )

    print(
        f"URL: {url}"
    )

    print(
        f"Destination: {destination}"
    )

    print()

    try:

        with urllib.request.urlopen(
            url
        ) as response:

            total = int(
                response.headers.get(
                    "Content-Length",
                    0,
                )
            )

            with destination.open(
                "wb"
            ) as output:

                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                ) as progress:

                    while True:

                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        output.write(
                            chunk
                        )

                        progress.update(
                            len(chunk)
                        )

    except Exception:

        if destination.exists():

            destination.unlink()

        raise


# =========================================================
# Archive validation
# =========================================================

def validate_archive(
    archive_path: Path,
):

    print()
    print(
        "Validating archive..."
    )

    if not archive_path.exists():

        raise FileNotFoundError(
            f"Archive not found:\n"
            f"{archive_path}"
        )

    try:

        with tarfile.open(
            archive_path,
            "r:gz",
        ) as archive:

            members = archive.getmembers()

            if not members:

                raise RuntimeError(
                    "LibriSpeech archive "
                    "contains no files."
                )

    except Exception as exc:

        raise RuntimeError(
            "LibriSpeech archive "
            "could not be opened. "
            "The download may be incomplete."
        ) from exc

    print(
        f"Archive OK "
        f"({len(members)} entries)"
    )


# =========================================================
# Extraction
# =========================================================

def extract_archive(
    archive_path: Path,
):

    if EXTRACT_DIR.exists():

        print()
        print(
            "LibriSpeech is already "
            "extracted."
        )

        return

    print()
    print(
        "Extracting LibriSpeech..."
    )

    EXTRACT_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as archive:

        archive.extractall(
            EXTRACT_DIR.parent
        )

    if not EXTRACT_DIR.exists():

        raise RuntimeError(
            "Extraction completed but "
            "LibriSpeech directory was "
            "not found."
        )

    print(
        f"Extracted to:\n"
        f"{EXTRACT_DIR}"
    )


# =========================================================
# Text normalization
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = text.lower().strip()

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

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern
        in WAKEWORD_PATTERNS
    )


# =========================================================
# LibriSpeech transcript parsing
# =========================================================

def load_transcripts():

    transcript_files = sorted(
        EXTRACT_DIR.rglob(
            "*.trans.txt"
        )
    )

    if not transcript_files:

        raise RuntimeError(
            "No LibriSpeech transcript "
            "files found."
        )

    entries = []

    for transcript_path in transcript_files:

        with transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(
                    " ",
                    1,
                )

                if len(parts) != 2:
                    continue

                utterance_id, text = parts

                wav_path = (
                    transcript_path.parent
                    / f"{utterance_id}.flac"
                )

                if not wav_path.exists():

                    continue

                if contains_wakeword(
                    text
                ):

                    continue

                entries.append(
                    {
                        "utterance_id":
                            utterance_id,
                        "text":
                            text,
                        "source":
                            wav_path,
                    }
                )

    if not entries:

        raise RuntimeError(
            "No usable LibriSpeech "
            "utterances were found."
        )

    return entries


# =========================================================
# Audio loading
# =========================================================

def load_audio(
    path: Path,
):

    try:

        import soundfile as sf

    except ImportError:

        raise RuntimeError(
            "soundfile is required.\n\n"
            "Install with:\n"
            "pip install soundfile"
        )

    audio, sample_rate = (
        sf.read(
            str(path),
            dtype="float32",
        )
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    if audio.ndim != 1:

        raise ValueError(
            f"Unsupported audio shape: "
            f"{audio.shape}"
        )

    return (
        audio,
        int(sample_rate),
    )


# =========================================================
# Resampling
# =========================================================

def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
):

    if (
        source_rate
        == target_rate
    ):

        return audio.astype(
            np.float32,
            copy=False,
        )

    target_length = int(
        round(
            len(audio)
            * target_rate
            / source_rate
        )
    )

    if target_length <= 1:

        return np.zeros(
            target_length,
            dtype=np.float32,
        )

    old_positions = np.linspace(
        0,
        1,
        len(audio),
        endpoint=False,
    )

    new_positions = np.linspace(
        0,
        1,
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
# Audio normalization
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
# Save WAV
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
        "ASTA LIBRISPEECH NEGATIVE DATASET"
    )
    print("=" * 50)

    print(
        "Dataset : LibriSpeech dev-clean"
    )

    print(
        f"Target  : {TARGET_SAMPLES}"
    )

    print(
        f"Output  : {OUTPUT_DIR}"
    )

    print("=" * 50)

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    download_file(
        LIBRISPEECH_URL,
        ARCHIVE_PATH,
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    validate_archive(
        ARCHIVE_PATH
    )

    # -----------------------------------------------------
    # Extract
    # -----------------------------------------------------

    extract_archive(
        ARCHIVE_PATH
    )

    # -----------------------------------------------------
    # Load transcripts
    # -----------------------------------------------------

    print()
    print(
        "Reading LibriSpeech "
        "transcripts..."
    )

    entries = load_transcripts()

    print(
        f"Usable utterances : "
        f"{len(entries)}"
    )

    # -----------------------------------------------------
    # Shuffle
    # -----------------------------------------------------

    random.shuffle(
        entries
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_files = list(
        OUTPUT_DIR.glob(
            "librispeech_*.wav"
        )
    )

    if existing_files:

        print()
        print(
            f"Found {len(existing_files)} "
            "existing LibriSpeech outputs."
        )

        print(
            "They will be reused when "
            "their manifest entries exist."
        )

    # -----------------------------------------------------
    # Counters
    # -----------------------------------------------------

    generated = 0
    duration_filtered = 0
    duplicates = 0
    conversion_errors = 0

    seen_hashes = set()

    manifest_rows = []

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    progress = tqdm(
        total=TARGET_SAMPLES,
        desc="Preparing negatives",
        unit="clip",
    )

    for entry in entries:

        if generated >= TARGET_SAMPLES:

            break

        source_path = (
            entry["source"]
        )

        try:

            audio, source_rate = (
                load_audio(
                    source_path
                )
            )

        except Exception:

            conversion_errors += 1

            continue

        duration = (
            len(audio)
            / source_rate
            if source_rate > 0
            else 0
        )

        if (
            duration
            < MIN_DURATION_SECONDS
            or
            duration
            > MAX_DURATION_SECONDS
        ):

            duration_filtered += 1

            continue

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

        digest = audio_hash(
            audio
        )

        if digest in seen_hashes:

            duplicates += 1

            continue

        seen_hashes.add(
            digest
        )

        filename = (
            f"librispeech_"
            f"{generated:06d}.wav"
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        try:

            save_wav(
                output_path,
                audio,
            )

        except Exception:

            conversion_errors += 1

            continue

        manifest_rows.append([
            filename,
            entry["utterance_id"],
            str(
                source_path
            ),
            entry["text"],
            f"{duration:.6f}",
            source_rate,
            TARGET_SAMPLE_RATE,
            digest,
        ])

        generated += 1

        progress.update(1)

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
            "utterance_id",
            "source_file",
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
        "LIBRISPEECH NEGATIVE DATASET COMPLETE"
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
        f"Duration filtered  : "
        f"{duration_filtered}"
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
            "The target number of clips "
            "was not reached."
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

        print(
            "LibriSpeech negative subset "
            "created successfully."
        )

    print("=" * 50)


if __name__ == "__main__":
    main()