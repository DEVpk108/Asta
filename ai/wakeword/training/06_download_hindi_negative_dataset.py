"""
06_download_hindi_negative_dataset.py

Build a Hindi negative dataset for ASTA.

Features:
- Resume existing Hindi negative dataset
- Preserves existing WAV files
- Preserves existing manifest entries
- Downloads metadata only first
- Downloads individual MP3 files on demand
- NEVER uses snapshot_download()
- Avoids downloading the entire dataset
- Skips already-used source files
- Converts audio to 22050 Hz mono WAV
- Filters by duration
- Detects duplicates using SHA256
- Continues until TARGET_SAMPLES is reached
- Safe to stop and rerun

Dataset:
    dhruvkys/11k-asr

Source:
    Mozilla Common Voice Hindi

Author: ASTA
"""

from pathlib import Path
import csv
import hashlib
import random
import shutil
import tempfile
import time

import numpy as np
import soundfile as sf
from tqdm import tqdm

from huggingface_hub import hf_hub_download


# ============================================================
# CONFIG
# ============================================================

DATASET_ID = "dhruvkys/11k-asr"

SPLIT = "train"

TARGET_SAMPLES = 2990

TARGET_SAMPLE_RATE = 22050

MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 8.0

# We need a lot more candidates because many clips can be
# rejected by duration / duplicate / download checks.
MAX_CANDIDATES_TO_CHECK = 8500

# Metadata is small, so cache it locally.
METADATA_FILENAME = "metadata.tsv"

# Existing source cache.
SOURCE_ROOT = (
    Path("ai")
    / "wakeword"
    / "generated"
    / "negative_sources"
    / "11k-asr"
)

METADATA_CACHE = SOURCE_ROOT / SPLIT / METADATA_FILENAME

# Final Hindi negative dataset.
OUTPUT_ROOT = (
    Path("ai")
    / "wakeword"
    / "generated"
    / "downloaded_negative"
    / "hindi"
)

MANIFEST_PATH = OUTPUT_ROOT / "hindi_manifest.csv"

# Temporary downloaded MP3 files.
AUDIO_CACHE = SOURCE_ROOT / SPLIT / "clips"

# Randomization makes repeated runs less biased toward the
# beginning of the metadata file.
RANDOMIZE_CANDIDATES = True

# Small pause between failed requests.
RETRY_DELAY_SECONDS = 1.5

MAX_DOWNLOAD_RETRIES = 3


# ============================================================
# DIRECTORY SETUP
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIO_CACHE.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def print_header():
    print("\n" + "=" * 50)
    print("ASTA HINDI NEGATIVE DATASET")
    print("=" * 50)
    print(f"Dataset : {DATASET_ID}")
    print(f"Split   : {SPLIT}")
    print(f"Target  : {TARGET_SAMPLES}")
    print(f"Sample rate : {TARGET_SAMPLE_RATE} Hz")
    print(f"Output  : {OUTPUT_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("=" * 50)
    print()


def sha256_file(path: Path) -> str:
    """
    Calculate SHA256 for a WAV file.
    """

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def get_existing_manifest():
    """
    Read existing manifest.

    Returns:
        rows
        used_source_files
        used_hashes
    """

    rows = []

    used_source_files = set()

    used_hashes = set()

    if not MANIFEST_PATH.exists():

        return (
            rows,
            used_source_files,
            used_hashes,
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

            source_file = (
                row.get("source_file")
                or row.get("source")
                or ""
            ).strip()

            if source_file:

                used_source_files.add(
                    source_file
                )

            file_hash = (
                row.get("sha256")
                or ""
            ).strip()

            if file_hash:

                used_hashes.add(
                    file_hash
                )

    return (
        rows,
        used_source_files,
        used_hashes,
    )


def rebuild_existing_hashes(
    rows,
    used_hashes,
):
    """
    If an older manifest did not contain hashes,
    calculate them from existing WAV files.
    """

    if used_hashes:

        return used_hashes

    print(
        "Calculating hashes for existing "
        "Hindi negatives..."
    )

    for row in rows:

        wav_path = row.get("wav_path", "").strip()

        if not wav_path:

            continue

        path = Path(wav_path)

        if not path.is_absolute():

            path = Path.cwd() / path

        if not path.exists():

            continue

        try:

            file_hash = sha256_file(path)

            used_hashes.add(file_hash)

        except Exception:

            pass

    return used_hashes


def count_existing_valid_files(rows):
    """
    Count existing WAVs that are still present.
    """

    count = 0

    for row in rows:

        wav_path = row.get(
            "wav_path",
            "",
        ).strip()

        if not wav_path:

            continue

        path = Path(wav_path)

        if not path.is_absolute():

            path = Path.cwd() / path

        if path.exists():

            count += 1

    return count


def write_manifest_header_if_needed():

    if MANIFEST_PATH.exists():

        return

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wav_path",
                "source_file",
                "text",
                "duration_seconds",
                "sample_rate",
                "sha256",
            ],
        )

        writer.writeheader()


def append_manifest_row(
    wav_path,
    source_file,
    text,
    duration_seconds,
    file_hash,
):

    write_manifest_header_if_needed()

    # Store relative paths where possible.
    try:

        relative_wav = wav_path.resolve().relative_to(
            Path.cwd().resolve()
        )

        wav_string = str(
            relative_wav
        ).replace("\\", "/")

    except ValueError:

        wav_string = str(
            wav_path
        ).replace("\\", "/")

    with MANIFEST_PATH.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wav_path",
                "source_file",
                "text",
                "duration_seconds",
                "sample_rate",
                "sha256",
            ],
        )

        writer.writerow(
            {
                "wav_path": wav_string,
                "source_file": source_file,
                "text": text,
                "duration_seconds": f"{duration_seconds:.4f}",
                "sample_rate": TARGET_SAMPLE_RATE,
                "sha256": file_hash,
            }
        )


# ============================================================
# METADATA
# ============================================================

def download_metadata():
    """
    Download only metadata.tsv.

    Does NOT download audio.
    """

    if METADATA_CACHE.exists():

        print(
            f"Using cached metadata:\n"
            f"{METADATA_CACHE}\n"
        )

        return METADATA_CACHE

    print("Downloading metadata only...")
    print(f"Dataset : {DATASET_ID}")
    print(f"File    : {SPLIT}/{METADATA_FILENAME}")
    print()

    downloaded_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"{SPLIT}/{METADATA_FILENAME}",
        repo_type="dataset",
        local_dir=str(SOURCE_ROOT),
    )

    downloaded_path = Path(
        downloaded_path
    )

    return downloaded_path


def read_metadata(metadata_path):
    """
    Read metadata.tsv.

    Expected columns:
        file_name
        text
    """

    rows = []

    with metadata_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            filename = (
                row.get("file_name")
                or row.get("path")
                or ""
            ).strip()

            text = (
                row.get("text")
                or ""
            ).strip()

            if not filename:

                continue

            rows.append(
                {
                    "file_name": filename,
                    "text": text,
                }
            )

    return rows


# ============================================================
# DOWNLOAD
# ============================================================

def download_audio(
    source_filename,
):
    """
    Download ONE MP3 file.

    Never downloads the entire dataset.
    """

    relative_path = (
        f"{SPLIT}/clips/{source_filename}"
    )

    destination = (
        AUDIO_CACHE / source_filename
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Already cached.
    if destination.exists():

        return destination

    last_error = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_RETRIES + 1,
    ):

        try:

            downloaded = hf_hub_download(
                repo_id=DATASET_ID,
                filename=relative_path,
                repo_type="dataset",
                local_dir=str(SOURCE_ROOT),
            )

            downloaded = Path(
                downloaded
            )

            if downloaded.exists():

                return downloaded

        except Exception as exc:

            last_error = exc

            if attempt < MAX_DOWNLOAD_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    raise RuntimeError(
        f"Failed downloading "
        f"{source_filename}: "
        f"{last_error}"
    )


# ============================================================
# AUDIO PROCESSING
# ============================================================

def process_audio(
    source_path,
    output_path,
):
    """
    Read MP3 and convert to:
        22050 Hz
        mono
        PCM16 WAV
    """

    import librosa

    audio, sample_rate = librosa.load(
        str(source_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    if audio is None:

        return None

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:

        return None

    # Remove NaN / Inf.
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    duration = (
        len(audio)
        / TARGET_SAMPLE_RATE
    )

    if duration < MIN_DURATION_SECONDS:

        return None

    if duration > MAX_DURATION_SECONDS:

        return None

    # Normalize only if necessary.
    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1.0:

        audio = (
            audio
            / peak
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )

    return duration


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # Existing dataset
    # --------------------------------------------------------

    (
        existing_rows,
        used_sources,
        used_hashes,
    ) = get_existing_manifest()

    existing_count = count_existing_valid_files(
        existing_rows
    )

    used_hashes = rebuild_existing_hashes(
        existing_rows,
        used_hashes,
    )

    remaining = max(
        0,
        TARGET_SAMPLES - existing_count,
    )

    print(
        f"Existing valid samples : "
        f"{existing_count}"
    )

    print(
        f"Target samples          : "
        f"{TARGET_SAMPLES}"
    )

    print(
        f"Remaining required      : "
        f"{remaining}"
    )

    print()

    if remaining == 0:

        print(
            "Target already reached."
        )

        print(
            "\nRESULT: PASS"
        )

        return

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = download_metadata()

    rows = read_metadata(
        metadata_path
    )

    print(
        f"Metadata rows : "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    candidates = [
        row
        for row in rows
        if row["file_name"]
        not in used_sources
    ]

    if RANDOMIZE_CANDIDATES:

        random.shuffle(
            candidates
        )

    candidates = candidates[
        :MAX_CANDIDATES_TO_CHECK
    ]

    print(
        f"New candidates available : "
        f"{len(candidates)}"
    )

    print()

    # --------------------------------------------------------
    # Output numbering
    # --------------------------------------------------------

    next_index = existing_count

    generated = 0
    candidates_checked = 0

    text_filtered = 0
    duration_filtered = 0
    duplicates = 0
    download_errors = 0
    conversion_errors = 0

    progress = tqdm(
        total=remaining,
        desc="Preparing Hindi negatives",
        unit="clip",
    )

    # --------------------------------------------------------
    # Process candidates
    # --------------------------------------------------------

    for candidate in candidates:

        if generated >= remaining:

            break

        candidates_checked += 1

        source_filename = (
            candidate["file_name"]
        )

        text = (
            candidate["text"]
        )

        # ----------------------------------------------------
        # Skip already-used source
        # ----------------------------------------------------

        if source_filename in used_sources:

            continue

        # ----------------------------------------------------
        # Download one MP3
        # ----------------------------------------------------

        try:

            source_path = download_audio(
                source_filename
            )

        except Exception:

            download_errors += 1

            continue

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        output_path = (
            OUTPUT_ROOT
            / f"hindi_{next_index:05d}.wav"
        )

        # ----------------------------------------------------
        # Convert
        # ----------------------------------------------------

        try:

            duration = process_audio(
                source_path,
                output_path,
            )

        except Exception:

            conversion_errors += 1

            if output_path.exists():

                try:
                    output_path.unlink()
                except Exception:
                    pass

            continue

        # ----------------------------------------------------
        # Duration rejection
        # ----------------------------------------------------

        if duration is None:

            duration_filtered += 1

            if output_path.exists():

                try:
                    output_path.unlink()
                except Exception:
                    pass

            continue

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        try:

            file_hash = sha256_file(
                output_path
            )

        except Exception:

            conversion_errors += 1

            try:
                output_path.unlink()
            except Exception:
                pass

            continue

        if file_hash in used_hashes:

            duplicates += 1

            try:
                output_path.unlink()
            except Exception:
                pass

            continue

        # ----------------------------------------------------
        # Successful sample
        # ----------------------------------------------------

        append_manifest_row(
            wav_path=output_path,
            source_file=source_filename,
            text=text,
            duration_seconds=duration,
            file_hash=file_hash,
        )

        used_sources.add(
            source_filename
        )

        used_hashes.add(
            file_hash
        )

        generated += 1
        next_index += 1

        progress.update(1)

    progress.close()

    # --------------------------------------------------------
    # Final count
    # --------------------------------------------------------

    final_count = (
        existing_count
        + generated
    )

    print()

    print("=" * 50)
    print("HINDI NEGATIVE DATASET COMPLETE")
    print("=" * 50)

    print(
        f"Existing preserved : "
        f"{existing_count}"
    )

    print(
        f"Newly generated     : "
        f"{generated}"
    )

    print(
        f"Generated total     : "
        f"{final_count}"
    )

    print(
        f"Target              : "
        f"{TARGET_SAMPLES}"
    )

    print(
        f"Candidates checked  : "
        f"{candidates_checked}"
    )

    print(
        f"Text filtered       : "
        f"{text_filtered}"
    )

    print(
        f"Duration filtered   : "
        f"{duration_filtered}"
    )

    print(
        f"Duplicates          : "
        f"{duplicates}"
    )

    print(
        f"Download errors     : "
        f"{download_errors}"
    )

    print(
        f"Conversion errors   : "
        f"{conversion_errors}"
    )

    print(
        f"Manifest            : "
        f"{MANIFEST_PATH}"
    )

    print("=" * 50)

    if final_count >= TARGET_SAMPLES:

        print()
        print("RESULT: PASS")
        print(
            "Hindi negative dataset target reached."
        )

    else:

        print()
        print("RESULT: INCOMPLETE")

        print(
            f"Still required: "
            f"{TARGET_SAMPLES - final_count}"
        )

        print(
            "Existing samples were preserved."
        )

        print(
            "Run this script again to continue."
        )

    print("=" * 50)


if __name__ == "__main__":

    main()