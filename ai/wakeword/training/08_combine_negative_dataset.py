"""
08_combine_negative_dataset.py

Combine validated Hindi + English negative datasets into one final
negative dataset for ASTA wake-word training.

Sources:
    generated/downloaded_negative/
        ├── hindi/
        │   ├── *.wav
        │   └── hindi_manifest.csv
        │
        └── *.wav
            └── librispeech_manifest.csv

Output:
    generated/negative/
        ├── *.wav
        └── negative_manifest.csv
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEGATIVE_ROOT = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "downloaded_negative"
)

HINDI_DIR = NEGATIVE_ROOT / "hindi"

# LibriSpeech files are produced directly inside downloaded_negative
ENGLISH_DIR = NEGATIVE_ROOT

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "negative"
)

OUTPUT_MANIFEST = OUTPUT_DIR / "negative_manifest.csv"


# ============================================================
# CONFIG
# ============================================================

HINDI_MANIFEST = HINDI_DIR / "hindi_manifest.csv"
ENGLISH_MANIFEST = NEGATIVE_ROOT / "librispeech_manifest.csv"

COPY_FILES = True
OVERWRITE = False


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {path}")

        return list(reader)


def find_wav(source_dir: Path, relative_path: str) -> Path:
    """
    Resolve a manifest path robustly.

    Handles manifests containing:
        foo.wav
        subdir/foo.wav
        /absolute/path/foo.wav
    """

    candidate = Path(relative_path)

    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = source_dir / candidate

    if direct.exists():
        return direct

    # Fallback: filename search.
    matches = list(source_dir.rglob(candidate.name))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous WAV reference '{relative_path}' "
            f"in {source_dir}"
        )

    raise FileNotFoundError(
        f"WAV not found: '{relative_path}' "
        f"(source: {source_dir})"
    )


def copy_wav(
    source: Path,
    destination: Path,
) -> None:

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():

        if not OVERWRITE:
            existing_hash = sha256_file(destination)
            source_hash = sha256_file(source)

            if existing_hash == source_hash:
                return

            raise FileExistsError(
                f"Destination already exists with different content:\n"
                f"{destination}"
            )

        destination.unlink()

    shutil.copy2(source, destination)


# ============================================================
# MANIFEST NORMALIZATION
# ============================================================

def normalize_entry(
    row: dict[str, str],
    source_name: str,
    source_manifest: Path,
    source_dir: Path,
) -> tuple[Path, dict[str, str]]:

    # Support the common manifest column names used in this project.
    wav_ref = (
        row.get("path")
        or row.get("wav")
        or row.get("wav_path")
        or row.get("file")
        or row.get("filename")
    )

    if not wav_ref:
        raise RuntimeError(
            f"Could not find WAV path column in {source_manifest}\n"
            f"Columns: {list(row.keys())}"
        )

    source_wav = find_wav(source_dir, wav_ref)

    source_hash = (
        row.get("sha256")
        or row.get("hash")
        or sha256_file(source_wav)
    )

    text = (
        row.get("text")
        or row.get("transcript")
        or ""
    )

    duration = (
        row.get("duration")
        or row.get("duration_sec")
        or ""
    )

    sample_rate = (
        row.get("sample_rate")
        or row.get("sample_rate_hz")
        or ""
    )

    original_sample_rate = (
        row.get("original_sample_rate")
        or row.get("source_sample_rate")
        or ""
    )

    speaker = row.get("speaker") or ""
    speaker_id = row.get("speaker_id") or ""

    return source_wav, {
        "path": "",
        "source": source_name,
        "source_file": source_wav.name,
        "text": text,
        "duration": duration,
        "sample_rate": sample_rate,
        "original_sample_rate": original_sample_rate,
        "speaker": speaker,
        "speaker_id": speaker_id,
        "sha256": source_hash,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 58)
    print("ASTA NEGATIVE DATASET COMBINATION")
    print("=" * 58)

    print(f"Hindi manifest   : {HINDI_MANIFEST}")
    print(f"English manifest : {ENGLISH_MANIFEST}")
    print(f"Output           : {OUTPUT_DIR}")
    print(f"Output manifest  : {OUTPUT_MANIFEST}")
    print("=" * 58)
    print()

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------

    if not HINDI_MANIFEST.exists():
        raise FileNotFoundError(
            f"Hindi manifest missing:\n{HINDI_MANIFEST}"
        )

    if not ENGLISH_MANIFEST.exists():
        raise FileNotFoundError(
            f"English manifest missing:\n{ENGLISH_MANIFEST}"
        )

    # --------------------------------------------------------
    # Read manifests
    # --------------------------------------------------------

    hindi_rows = read_manifest(HINDI_MANIFEST)
    english_rows = read_manifest(ENGLISH_MANIFEST)

    print(f"Hindi entries     : {len(hindi_rows)}")
    print(f"English entries   : {len(english_rows)}")
    print(f"Expected combined : {len(hindi_rows) + len(english_rows)}")
    print()

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep sources separated inside final dataset.
    hindi_output = OUTPUT_DIR / "hindi"
    english_output = OUTPUT_DIR / "english"

    hindi_output.mkdir(parents=True, exist_ok=True)
    english_output.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, str]] = []

    seen_hashes: set[str] = set()

    duplicate_count = 0
    copied_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    datasets = [
        (
            "hindi",
            hindi_rows,
            HINDI_MANIFEST,
            HINDI_DIR,
            hindi_output,
        ),
        (
            "english",
            english_rows,
            ENGLISH_MANIFEST,
            ENGLISH_DIR,
            english_output,
        ),
    ]

    for (
        source_name,
        rows,
        manifest_path,
        source_dir,
        destination_dir,
    ) in datasets:

        print(f"Processing {source_name} negatives...")

        for index, row in enumerate(rows):

            source_wav, normalized = normalize_entry(
                row=row,
                source_name=source_name,
                source_manifest=manifest_path,
                source_dir=source_dir,
            )

            file_hash = normalized["sha256"]

            # ------------------------------------------------
            # Duplicate detection
            # ------------------------------------------------

            if file_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(file_hash)

            # ------------------------------------------------
            # Generate deterministic output filename
            # ------------------------------------------------

            output_name = (
                f"{source_name}_{index:05d}.wav"
            )

            destination = destination_dir / output_name

            copy_wav(
                source=source_wav,
                destination=destination,
            )

            normalized["path"] = (
                f"{source_name}/{output_name}"
            )

            combined_rows.append(normalized)

            copied_count += 1

            if (
                (index + 1) % 500 == 0
                or index + 1 == len(rows)
            ):
                print(
                    f"  {source_name}: "
                    f"{index + 1}/{len(rows)}"
                )

        print()

    # --------------------------------------------------------
    # Write combined manifest
    # --------------------------------------------------------

    fieldnames = [
        "path",
        "source",
        "source_file",
        "text",
        "duration",
        "sample_rate",
        "original_sample_rate",
        "speaker",
        "speaker_id",
        "sha256",
    ]

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(combined_rows)

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    expected = len(hindi_rows) + len(english_rows)
    actual = len(combined_rows)

    output_wavs = list(OUTPUT_DIR.rglob("*.wav"))

    print("=" * 58)
    print("NEGATIVE DATASET COMBINATION COMPLETE")
    print("=" * 58)

    print(f"Hindi source       : {len(hindi_rows)}")
    print(f"English source     : {len(english_rows)}")
    print(f"Expected total     : {expected}")
    print(f"Combined samples   : {actual}")
    print(f"Output WAV files   : {len(output_wavs)}")
    print(f"Duplicates removed : {duplicate_count}")
    print(f"Copied             : {copied_count}")
    print(f"Skipped            : {skipped_count}")
    print(f"Manifest           : {OUTPUT_MANIFEST}")
    print("=" * 58)

    if actual != len(output_wavs):
        print()
        print("RESULT: FAIL")
        print(
            "Manifest/WAV count mismatch. "
            "Run validation before continuing."
        )
        raise SystemExit(1)

    if duplicate_count:
        print()
        print(
            "NOTE: Duplicate audio files were removed "
            "using SHA-256."
        )

    print()
    print("RESULT: PASS")
    print("Combined negative dataset is ready for validation.")
    print("=" * 58)


if __name__ == "__main__":
    main()
    