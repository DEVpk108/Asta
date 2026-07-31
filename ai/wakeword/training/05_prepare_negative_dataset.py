from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ============================================================
# ASTA LIBRISPEECH ENGLISH NEGATIVE DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_ROOT = PROJECT_ROOT / "ai" / "wakeword" / "generated"

OUTPUT_DIR = GENERATED_ROOT / "downloaded_negative"
SOURCE_DIR = GENERATED_ROOT / "negative_sources"

MANIFEST_PATH = OUTPUT_DIR / "librispeech_manifest.csv"

DEV_CLEAN_ARCHIVE = SOURCE_DIR / "dev-clean.tar.gz"
DEV_OTHER_ARCHIVE = SOURCE_DIR / "dev-other.tar.gz"

EXTRACT_ROOT = SOURCE_DIR / "LibriSpeech"

DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DEV_OTHER_URL = "https://www.openslr.org/resources/12/dev-other.tar.gz"

TARGET_TOTAL = 3000
TARGET_SAMPLE_RATE = 22050

MIN_DURATION = 0.5
MAX_DURATION = 10.0

# We intentionally use dev-clean first and dev-other only for
# whatever remains after dev-clean is exhausted.
PREFERRED_SPLITS = [
    ("dev-clean", DEV_CLEAN_ARCHIVE, DEV_CLEAN_URL),
    ("dev-other", DEV_OTHER_ARCHIVE, DEV_OTHER_URL),
]

MANIFEST_FIELDS = [
    "filename",
    "source_file",
    "utterance_id",
    "text",
    "source_duration_seconds",
    "source_sample_rate",
    "output_sample_rate",
    "sha256",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print()
        print("Archive already exists.")
        print(f"Using: {destination}")
        return

    print()
    print("Downloading archive...")
    print(f"URL: {url}")
    print(f"Destination: {destination}")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ASTA-WakeWord-Dataset-Builder/1.0"
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None

        temp_path = destination.with_suffix(destination.suffix + ".part")

        downloaded = 0

        with open(temp_path, "wb") as output:
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)
                    progress.update(len(chunk))

        if downloaded == 0:
            raise RuntimeError("Downloaded archive is empty.")

        temp_path.replace(destination)

    print("Download complete.")


# ============================================================
# ARCHIVE VALIDATION
# ============================================================

def validate_archive(archive_path: Path) -> int:
    print()
    print("Validating archive...")

    if not archive_path.exists():
        raise FileNotFoundError(str(archive_path))

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()

            if not members:
                raise RuntimeError("Archive contains no entries.")

            # Basic path traversal protection.
            for member in members:
                member_path = Path(member.name)

                if member_path.is_absolute():
                    raise RuntimeError(
                        f"Unsafe absolute path in archive: {member.name}"
                    )

                if ".." in member_path.parts:
                    raise RuntimeError(
                        f"Unsafe traversal path in archive: {member.name}"
                    )

            count = len(members)

    except tarfile.TarError as exc:
        raise RuntimeError(
            f"Invalid/corrupt tar archive: {archive_path}"
        ) from exc

    print(f"Archive OK ({count} entries)")
    return count


# ============================================================
# EXTRACTION
# ============================================================

def split_root(split_name: str) -> Path:
    return EXTRACT_ROOT / "LibriSpeech" / split_name


def is_extracted(split_name: str) -> bool:
    root = split_root(split_name)
    return root.exists() and root.is_dir()


def extract_archive(split_name: str, archive_path: Path) -> Path:
    destination = EXTRACT_ROOT

    if is_extracted(split_name):
        print()
        print(f"LibriSpeech {split_name} already extracted.")
        return split_root(split_name)

    print()
    print(f"Extracting LibriSpeech {split_name}...")

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()

        base = destination.resolve()

        for member in members:
            target = (destination / member.name).resolve()

            if not str(target).startswith(str(base) + os.sep):
                raise RuntimeError(
                    f"Unsafe archive member: {member.name}"
                )

        tar.extractall(destination)

    if not is_extracted(split_name):
        raise RuntimeError(
            f"Extraction completed but expected directory was not found:\n"
            f"{split_root(split_name)}"
        )

    print(f"Extracted to:\n{split_root(split_name)}")

    return split_root(split_name)


# ============================================================
# EXISTING MANIFEST HANDLING
# ============================================================

def read_existing_manifest() -> List[Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []

    rows: List[Dict[str, str]] = []

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            return []

        for raw in reader:
            row = {
                field: raw.get(field, "")
                for field in MANIFEST_FIELDS
            }

            # Preserve useful values from alternative older schemas.
            if not row["filename"]:
                row["filename"] = raw.get("file", "")

            if not row["source_file"]:
                row["source_file"] = raw.get("source", "")

            if not row["text"]:
                row["text"] = raw.get("transcript", "")

            if not row["utterance_id"]:
                row["utterance_id"] = raw.get("id", "")

            rows.append(row)

    return rows


def existing_wav_files() -> Set[str]:
    if not OUTPUT_DIR.exists():
        return set()

    return {
        p.name.lower()
        for p in OUTPUT_DIR.glob("*.wav")
        if p.is_file()
    }


def normalize_existing_rows(
    rows: List[Dict[str, str]]
) -> List[Dict[str, str]]:

    normalized: List[Dict[str, str]] = []

    wavs = existing_wav_files()

    for row in rows:
        filename = row.get("filename", "").strip()

        if not filename:
            continue

        # Don't preserve manifest entries for missing files.
        if filename.lower() not in wavs:
            continue

        normalized.append(
            {
                "filename": filename,
                "source_file": row.get("source_file", ""),
                "utterance_id": row.get("utterance_id", ""),
                "text": row.get("text", ""),
                "source_duration_seconds": row.get(
                    "source_duration_seconds", ""
                ),
                "source_sample_rate": row.get(
                    "source_sample_rate", ""
                ),
                "output_sample_rate": row.get(
                    "output_sample_rate",
                    str(TARGET_SAMPLE_RATE),
                ),
                "sha256": row.get("sha256", ""),
            }
        )

    return normalized


# ============================================================
# EXISTING SOURCE IDENTIFIERS
# ============================================================

def existing_source_keys(
    rows: Iterable[Dict[str, str]]
) -> Set[str]:

    keys: Set[str] = set()

    for row in rows:
        source_file = row.get("source_file", "").strip()
        utterance_id = row.get("utterance_id", "").strip()

        if source_file:
            keys.add(source_file.lower())

        if utterance_id:
            keys.add(utterance_id.lower())

    return keys


# ============================================================
# LIBRISPEECH TRANSCRIPTS
# ============================================================

def read_transcripts(
    split_root_path: Path,
) -> Dict[str, str]:

    transcripts: Dict[str, str] = {}

    for trans_path in split_root_path.rglob("*.trans.txt"):
        with open(
            trans_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:

            for line in handle:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    continue

                utterance_id, text = parts

                transcripts[utterance_id.strip()] = text.strip()

    return transcripts


# ============================================================
# AUDIO HELPERS
# ============================================================

def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def read_audio(
    path: Path,
) -> Tuple[np.ndarray, int]:

    data, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = np.asarray(data, dtype=np.float32)

    return data, int(sample_rate)


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:

    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio.astype(np.float32)

    target_length = int(
        round(len(audio) * target_rate / source_rate)
    )

    if target_length <= 1:
        return np.asarray(audio[:1], dtype=np.float32)

    old_positions = np.linspace(
        0.0,
        1.0,
        num=len(audio),
        endpoint=False,
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        num=target_length,
        endpoint=False,
    )

    resampled = np.interp(
        new_positions,
        old_positions,
        audio,
    )

    return np.asarray(resampled, dtype=np.float32)


def write_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:

    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def collect_candidates(
    split_name: str,
    split_root_path: Path,
    already_used: Set[str],
) -> List[Tuple[Path, str, str]]:

    transcripts = read_transcripts(split_root_path)

    candidates: List[Tuple[Path, str, str]] = []

    for flac_path in sorted(split_root_path.rglob("*.flac")):
        utterance_id = flac_path.stem

        key = utterance_id.lower()

        if key in already_used:
            continue

        text = transcripts.get(utterance_id, "")

        try:
            duration = audio_duration(flac_path)
        except Exception:
            continue

        if duration < MIN_DURATION:
            continue

        if duration > MAX_DURATION:
            continue

        candidates.append(
            (
                flac_path,
                utterance_id,
                text,
            )
        )

    print(
        f"Usable LibriSpeech utterances "
        f"({split_name}) : {len(candidates)}"
    )

    return candidates


# ============================================================
# MANIFEST WRITER
# ============================================================

def write_manifest(
    rows: List[Dict[str, str]]
) -> None:

    temp_path = MANIFEST_PATH.with_suffix(
        MANIFEST_PATH.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            normalized = {
                field: row.get(field, "")
                for field in MANIFEST_FIELDS
            }

            writer.writerow(normalized)

    temp_path.replace(MANIFEST_PATH)


# ============================================================
# CREATE NEGATIVE
# ============================================================

def create_negative(
    source_path: Path,
    utterance_id: str,
    text: str,
    output_path: Path,
) -> Dict[str, str]:

    audio, source_rate = read_audio(source_path)

    source_duration = (
        len(audio) / source_rate
        if source_rate > 0
        else 0.0
    )

    if source_duration < MIN_DURATION:
        raise ValueError("Audio is too short.")

    if source_duration > MAX_DURATION:
        raise ValueError("Audio is too long.")

    audio = resample_audio(
        audio,
        source_rate,
        TARGET_SAMPLE_RATE,
    )

    write_wav(
        output_path,
        audio,
        TARGET_SAMPLE_RATE,
    )

    return {
        "filename": output_path.name,
        "source_file": str(source_path),
        "utterance_id": utterance_id,
        "text": text,
        "source_duration_seconds": f"{source_duration:.6f}",
        "source_sample_rate": str(source_rate),
        "output_sample_rate": str(TARGET_SAMPLE_RATE),
        "sha256": sha256_file(output_path),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 58)
    print("ASTA LIBRISPEECH ENGLISH NEGATIVE DATASET")
    print("=" * 58)
    print(f"Target total : {TARGET_TOTAL}")
    print(f"Sample rate  : {TARGET_SAMPLE_RATE} Hz")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Manifest     : {MANIFEST_PATH}")
    print("=" * 58)

    ensure_directories()

    # --------------------------------------------------------
    # Existing data
    # --------------------------------------------------------

    raw_existing = read_existing_manifest()
    existing_rows = normalize_existing_rows(raw_existing)

    wav_count = len(existing_wav_files())

    print()
    print("Existing English dataset")
    print(f"Manifest entries : {len(existing_rows)}")
    print(f"WAV files        : {wav_count}")

    # Reconcile filesystem and manifest.
    if len(existing_rows) != wav_count:
        print()
        print(
            "WARNING: Existing manifest/filesystem counts differ."
        )
        print(
            "Only existing WAV files represented by the manifest "
            "will be preserved."
        )

    current_rows = existing_rows

    # Remove duplicate filenames from existing manifest.
    seen_filenames: Set[str] = set()
    deduped_rows: List[Dict[str, str]] = []

    for row in current_rows:
        key = row["filename"].lower()

        if key in seen_filenames:
            continue

        seen_filenames.add(key)
        deduped_rows.append(row)

    current_rows = deduped_rows

    current_total = len(current_rows)

    remaining = TARGET_TOTAL - current_total

    print(f"Current total     : {current_total}")
    print(f"Remaining needed  : {max(remaining, 0)}")

    if current_total > TARGET_TOTAL:
        raise RuntimeError(
            f"Existing English dataset already contains "
            f"{current_total} samples, exceeding target "
            f"{TARGET_TOTAL}."
        )

    if remaining == 0:
        write_manifest(current_rows)

        print()
        print("=" * 58)
        print("ENGLISH NEGATIVE DATASET ALREADY COMPLETE")
        print("=" * 58)
        print(f"Final total : {len(current_rows)}")
        print(f"Manifest    : {MANIFEST_PATH}")
        print()
        print("RESULT: PASS")
        print("=" * 58)
        return

    # --------------------------------------------------------
    # Existing source IDs
    # --------------------------------------------------------

    used_sources = existing_source_keys(current_rows)

    # Also prevent reusing a source that was already generated
    # under the same output directory.
    generated_source_names = {
        row.get("source_file", "").lower()
        for row in current_rows
        if row.get("source_file")
    }

    used_sources.update(generated_source_names)

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    generated_new = 0
    duplicate_count = 0
    conversion_errors = 0
    duration_filtered = 0

    for split_name, archive_path, url in PREFERRED_SPLITS:

        if generated_new >= remaining:
            break

        # Download only when needed.
        download_file(
            url,
            archive_path,
        )

        validate_archive(archive_path)

        split_root_path = extract_archive(
            split_name,
            archive_path,
        )

        candidates = collect_candidates(
            split_name,
            split_root_path,
            used_sources,
        )

        if not candidates:
            continue

        needed_from_split = remaining - generated_new

        print()
        print(
            f"Preparing additional English negatives "
            f"from {split_name}: "
            f"0/{needed_from_split}"
        )

        progress = tqdm(
            candidates,
            total=min(
                len(candidates),
                needed_from_split,
            ),
            desc=f"Preparing {split_name}",
            unit="clip",
        )

        for source_path, utterance_id, text in progress:

            if generated_new >= remaining:
                break

            source_key = utterance_id.lower()

            if source_key in used_sources:
                duplicate_count += 1
                continue

            # Make the final filename globally unique.
            output_index = current_total + generated_new

            output_filename = (
                f"english_{output_index:06d}.wav"
            )

            output_path = OUTPUT_DIR / output_filename

            if output_path.exists():
                # Find next free name.
                while output_path.exists():
                    output_index += 1
                    output_filename = (
                        f"english_{output_index:06d}.wav"
                    )
                    output_path = (
                        OUTPUT_DIR / output_filename
                    )

            try:
                row = create_negative(
                    source_path=source_path,
                    utterance_id=utterance_id,
                    text=text,
                    output_path=output_path,
                )

            except ValueError:
                duration_filtered += 1
                continue

            except Exception as exc:
                conversion_errors += 1

                print()
                print(
                    f"WARNING: Failed to convert "
                    f"{source_path.name}: {exc}"
                )

                continue

            current_rows.append(row)

            used_sources.add(source_key)

            generated_new += 1

            progress.set_postfix(
                total=current_total + generated_new
            )

        progress.close()

    # --------------------------------------------------------
    # Final manifest
    # --------------------------------------------------------

    write_manifest(current_rows)

    final_total = len(current_rows)

    print()
    print("=" * 58)
    print("LIBRISPEECH ENGLISH NEGATIVE DATASET COMPLETE")
    print("=" * 58)
    print(f"Existing preserved : {current_total}")
    print(f"Newly generated    : {generated_new}")
    print(f"Final total        : {final_total}")
    print(f"Target             : {TARGET_TOTAL}")
    print(f"Duplicates skipped : {duplicate_count}")
    print(f"Duration filtered  : {duration_filtered}")
    print(f"Conversion errors  : {conversion_errors}")
    print(f"Manifest           : {MANIFEST_PATH}")
    print("=" * 58)

    if final_total == TARGET_TOTAL:
        print()
        print("RESULT: PASS")
        print("English negative dataset target reached.")
        print("=" * 58)
        return

    print()
    print("RESULT: INCOMPLETE")
    print(
        f"Target was not reached. "
        f"Missing {TARGET_TOTAL - final_total} samples."
    )
    print("=" * 58)

    raise SystemExit(1)


if __name__ == "__main__":
    main()