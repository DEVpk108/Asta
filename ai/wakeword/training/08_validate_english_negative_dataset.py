"""
ASTA English Negative Dataset Validator
=======================================

Validates the English negative dataset produced by
05_prepare_negative_dataset.py.

Expected layout:

ai/wakeword/generated/downloaded_negative/
    librispeech_manifest.csv
    *.wav

The validator intentionally does not require a particular manifest schema.
It discovers the filename/path column and validates the actual WAV files.

Run:
    python -m ai.wakeword.training.08_validate_english_negative_dataset
"""

from __future__ import annotations

import csv
import hashlib
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH_BYTES = 2  # PCM16

# Wake-word negative clips should normally be short.
MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = 15.0

# Silence threshold based on normalized int16 amplitude.
SILENCE_RMS_THRESHOLD = 0.001

# Warning thresholds.
CLIPPING_RATIO_WARNING = 0.0001  # 0.01%
MIN_NONZERO_RATIO = 0.001

# Manifest/output locations relative to this file:
# .../ai/wakeword/training/08_*.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATED_ROOT = PROJECT_ROOT / "ai" / "wakeword" / "generated"
DATASET_DIR = GENERATED_ROOT / "downloaded_negative"
MANIFEST_PATH = DATASET_DIR / "librispeech_manifest.csv"
REPORT_PATH = DATASET_DIR / "librispeech_validation_report.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def find_manifest_filename_column(fieldnames: list[str]) -> str | None:
    """Find the most likely filename/path column in a mixed-schema manifest."""
    preferred = (
        "filename",
        "file",
        "wav_file",
        "wav_filename",
        "output_filename",
        "output_file",
        "path",
        "filepath",
        "file_path",
    )

    lower_map = {name.strip().lower(): name for name in fieldnames}

    for candidate in preferred:
        if candidate in lower_map:
            return lower_map[candidate]

    # Fallback: any column whose name strongly suggests a file path.
    for original in fieldnames:
        name = original.strip().lower()
        if any(token in name for token in ("filename", "filepath", "file_path", "wav")):
            return original

    return None


def normalize_manifest_filename(value: str) -> str:
    """
    Normalize a manifest file reference.

    Handles:
    - absolute paths
    - Windows separators
    - paths beginning with the dataset directory
    - plain filenames
    """
    value = (value or "").strip().strip('"')
    value = value.replace("\\", "/")

    if not value:
        return ""

    # Keep only the filename if the manifest contains an absolute/external path.
    # This is deliberate: the validator's source of truth is DATASET_DIR.
    return Path(value).name


def discover_wavs() -> list[Path]:
    if not DATASET_DIR.exists():
        return []

    return sorted(
        p for p in DATASET_DIR.glob("*.wav")
        if p.is_file()
    )


def read_manifest() -> tuple[list[dict[str, str]], list[str], str | None]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError("Manifest has no header.")

        rows = list(reader)
        filename_column = find_manifest_filename_column(reader.fieldnames)

    return rows, reader.fieldnames, filename_column


def pcm16_metrics(raw: bytes) -> dict[str, float | int]:
    """Calculate basic signal metrics without requiring numpy/librosa."""
    if len(raw) < 2:
        return {
            "sample_count": 0,
            "max_abs": 0,
            "clipped_samples": 0,
            "clipping_ratio": 0.0,
            "rms": 0.0,
            "nonzero_ratio": 0.0,
        }

    usable = len(raw) - (len(raw) % 2)
    sample_count = usable // 2

    clipped = 0
    nonzero = 0
    sum_squares = 0.0
    max_abs = 0

    # PCM16 little-endian.
    for i in range(0, usable, 2):
        value = int.from_bytes(raw[i:i + 2], byteorder="little", signed=True)
        abs_value = abs(value)

        if abs_value > max_abs:
            max_abs = abs_value

        if value in (-32768, 32767):
            clipped += 1

        if value != 0:
            nonzero += 1

        normalized = value / 32768.0
        sum_squares += normalized * normalized

    rms = math.sqrt(sum_squares / sample_count)

    return {
        "sample_count": sample_count,
        "max_abs": max_abs,
        "clipped_samples": clipped,
        "clipping_ratio": clipped / sample_count if sample_count else 0.0,
        "rms": rms,
        "nonzero_ratio": nonzero / sample_count if sample_count else 0.0,
    }


def validate_wav(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "valid": True,
        "errors": [],
        "warnings": [],
        "duration": 0.0,
        "sample_rate": None,
        "channels": None,
        "sample_width": None,
        "frames": 0,
        "clipped_samples": 0,
        "clipping_ratio": 0.0,
        "max_abs": 0,
        "rms": 0.0,
    }

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype()

            result["channels"] = channels
            result["sample_rate"] = sample_rate
            result["sample_width"] = sample_width
            result["frames"] = frames

            if compression != "NONE":
                result["errors"].append(
                    f"compressed WAV ({compression})"
                )

            if sample_rate != TARGET_SAMPLE_RATE:
                result["errors"].append(
                    f"sample rate {sample_rate} Hz != {TARGET_SAMPLE_RATE} Hz"
                )

            if channels != EXPECTED_CHANNELS:
                result["errors"].append(
                    f"{channels} channels != {EXPECTED_CHANNELS}"
                )

            if sample_width != EXPECTED_SAMPLE_WIDTH_BYTES:
                result["errors"].append(
                    f"sample width {sample_width} bytes != "
                    f"{EXPECTED_SAMPLE_WIDTH_BYTES} bytes (PCM16)"
                )

            if sample_rate > 0:
                duration = frames / sample_rate
            else:
                duration = 0.0

            result["duration"] = duration

            if duration < MIN_DURATION_SECONDS:
                result["errors"].append(
                    f"duration {duration:.3f}s < {MIN_DURATION_SECONDS:.2f}s"
                )
            elif duration > MAX_DURATION_SECONDS:
                result["warnings"].append(
                    f"long clip {duration:.3f}s > {MAX_DURATION_SECONDS:.2f}s"
                )

            # Signal analysis is meaningful only for PCM16.
            if sample_width == EXPECTED_SAMPLE_WIDTH_BYTES and channels == EXPECTED_CHANNELS:
                raw = wav.readframes(frames)
                metrics = pcm16_metrics(raw)

                result["clipped_samples"] = int(metrics["clipped_samples"])
                result["clipping_ratio"] = float(metrics["clipping_ratio"])
                result["max_abs"] = int(metrics["max_abs"])
                result["rms"] = float(metrics["rms"])

                if result["clipped_samples"] > 0:
                    if result["clipping_ratio"] >= CLIPPING_RATIO_WARNING:
                        result["warnings"].append(
                            f"clipping ratio {result['clipping_ratio'] * 100:.4f}%"
                        )

                if result["rms"] <= SILENCE_RMS_THRESHOLD:
                    result["warnings"].append(
                        f"near-silent audio (RMS={result['rms']:.6f})"
                    )

                if metrics["nonzero_ratio"] < MIN_NONZERO_RATIO:
                    result["warnings"].append(
                        "almost all samples are zero"
                    )

    except (wave.Error, OSError, EOFError) as exc:
        result["valid"] = False
        result["errors"].append(f"unreadable WAV: {exc}")

    if result["errors"]:
        result["valid"] = False

    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "None"

    return "\n".join(
        f"  {key}: {value}"
        for key, value in counter.most_common()
    )


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> int:
    print_header("ASTA ENGLISH NEGATIVE DATASET VALIDATION")

    print(f"Dataset : {DATASET_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print()

    try:
        rows, fieldnames, filename_column = read_manifest()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    wav_files = discover_wavs()

    print(f"Manifest entries : {len(rows)}")
    print(f"WAV files        : {len(wav_files)}")
    print()

    if filename_column is None:
        print("ERROR: Could not identify filename column in manifest.")
        print(f"Manifest columns: {', '.join(fieldnames)}")
        return 1

    # ------------------------------------------------------------------
    # Manifest/filesystem mapping
    # ------------------------------------------------------------------

    manifest_names: list[str] = []

    for row in rows:
        manifest_names.append(
            normalize_manifest_filename(row.get(filename_column, ""))
        )

    manifest_name_set = {name for name in manifest_names if name}
    wav_name_set = {p.name for p in wav_files}

    missing_files = sorted(manifest_name_set - wav_name_set)
    unexpected_files = sorted(wav_name_set - manifest_name_set)

    duplicate_manifest_names = sorted(
        name for name, count in Counter(manifest_names).items()
        if name and count > 1
    )

    blank_manifest_names = sum(
        1 for name in manifest_names if not name
    )

    if missing_files:
        print(f"Missing WAV files : {len(missing_files)}")

    if unexpected_files:
        print(f"Unexpected WAVs   : {len(unexpected_files)}")

    if not missing_files and not unexpected_files:
        print("Filesystem/manifest mapping : OK")

    print()

    # ------------------------------------------------------------------
    # Per-file validation
    # ------------------------------------------------------------------

    results: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    clipping_warnings = 0
    duplicate_warnings = 0
    silent_clips = 0
    duration_warnings = 0
    format_warnings = 0

    # Only validate files represented in the manifest for the main result.
    files_to_check = [
        DATASET_DIR / name
        for name in manifest_names
        if name and (DATASET_DIR / name).exists()
    ]

    # Remove duplicate filesystem paths while retaining order.
    seen_paths: set[Path] = set()
    unique_files_to_check: list[Path] = []

    for path in files_to_check:
        resolved = path.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            unique_files_to_check.append(path)

    total = len(unique_files_to_check)

    for index, path in enumerate(unique_files_to_check, start=1):
        result = validate_wav(path)
        results.append(result)

        for error in result["errors"]:
            error_counts[error] += 1

        for warning in result["warnings"]:
            warning_counts[warning] += 1

            if "clipping ratio" in warning:
                clipping_warnings += 1
            if "near-silent" in warning or "almost all samples" in warning:
                silent_clips += 1
            if "long clip" in warning:
                duration_warnings += 1

        if any(
            "sample rate" in error
            or "channels" in error
            or "sample width" in error
            or "compressed WAV" in error
            for error in result["errors"]
        ):
            format_warnings += 1

        if index % 100 == 0 or index == total:
            print(f"Checked {index}/{total}...")

    # ------------------------------------------------------------------
    # Duplicate audio detection
    # ------------------------------------------------------------------

    hashes: dict[str, list[str]] = {}
    duplicate_groups = 0
    duplicate_files = 0

    for path in unique_files_to_check:
        try:
            digest = sha256_file(path)
        except OSError as exc:
            error_counts[f"hash error: {exc}"] += 1
            continue

        hashes.setdefault(digest, []).append(path.name)

    for names in hashes.values():
        if len(names) > 1:
            duplicate_groups += 1
            duplicate_files += len(names)
            duplicate_warnings += len(names) - 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    valid_count = sum(1 for result in results if result["valid"])

    total_clipped_samples = sum(
        int(result["clipped_samples"])
        for result in results
    )

    max_clipped_samples = max(
        (int(result["clipped_samples"]) for result in results),
        default=0,
    )

    max_clipping_ratio = max(
        (float(result["clipping_ratio"]) for result in results),
        default=0.0,
    )

    mapping_errors = len(missing_files) + len(unexpected_files)
    manifest_errors = (
        len(duplicate_manifest_names) + blank_manifest_names
    )

    total_errors = (
        len(error_counts)
        + mapping_errors
        + manifest_errors
    )

    # Error count should represent problematic records rather than
    # number of unique error messages.
    record_error_count = (
        sum(
            1 for result in results
            if result["errors"]
        )
        + len(missing_files)
        + len(unexpected_files)
        + len(duplicate_manifest_names)
        + blank_manifest_names
    )

    total_warnings = (
        sum(
            len(result["warnings"])
            for result in results
        )
        + duplicate_warnings
    )

    pass_result = (
        len(rows) == len(wav_files)
        and not missing_files
        and not unexpected_files
        and not duplicate_manifest_names
        and blank_manifest_names == 0
        and valid_count == len(rows)
        and record_error_count == 0
    )

    print_header("ENGLISH NEGATIVE VALIDATION COMPLETE")

    print(f"WAV files : {len(wav_files)}")
    print(f"Manifest  : {len(rows)}")
    print(f"Valid     : {valid_count}")
    print(f"Errors    : {record_error_count}")
    print(f"Warnings  : {total_warnings}")
    print()
    print(f"Clipping warnings : {clipping_warnings}")
    print(f"Duplicate warnings: {duplicate_warnings}")
    print(f"Silent clips      : {silent_clips}")
    print(f"Duration warnings : {duration_warnings}")
    print(f"Format warnings   : {format_warnings}")
    print(f"Missing files     : {len(missing_files)}")
    print(f"Unexpected files  : {len(unexpected_files)}")
    print(f"Max clipped samples: {max_clipped_samples}")
    print(f"Max clipping ratio: {max_clipping_ratio * 100:.4f}%")
    print()
    print(f"Report    : {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Write detailed report
    # ------------------------------------------------------------------

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as report:
        report.write("ASTA ENGLISH NEGATIVE DATASET VALIDATION REPORT\n")
        report.write("=" * 58 + "\n\n")

        report.write(f"Dataset: {DATASET_DIR}\n")
        report.write(f"Manifest: {MANIFEST_PATH}\n")
        report.write(f"Manifest filename column: {filename_column}\n\n")

        report.write("SUMMARY\n")
        report.write("-" * 58 + "\n")
        report.write(f"Manifest entries: {len(rows)}\n")
        report.write(f"WAV files: {len(wav_files)}\n")
        report.write(f"Valid: {valid_count}\n")
        report.write(f"Errors: {record_error_count}\n")
        report.write(f"Warnings: {total_warnings}\n")
        report.write(f"Missing files: {len(missing_files)}\n")
        report.write(f"Unexpected files: {len(unexpected_files)}\n")
        report.write(
            f"Max clipped samples: {max_clipped_samples}\n"
        )
        report.write(
            f"Max clipping ratio: {max_clipping_ratio * 100:.4f}%\n"
        )
        report.write("\n")

        report.write("MANIFEST SCHEMA\n")
        report.write("-" * 58 + "\n")
        report.write(", ".join(fieldnames) + "\n\n")

        if duplicate_manifest_names:
            report.write("DUPLICATE MANIFEST FILENAMES\n")
            report.write("-" * 58 + "\n")
            for name in duplicate_manifest_names:
                report.write(f"{name}\n")
            report.write("\n")

        if blank_manifest_names:
            report.write(
                f"Blank manifest filename entries: {blank_manifest_names}\n\n"
            )

        if missing_files:
            report.write("MISSING WAV FILES\n")
            report.write("-" * 58 + "\n")
            for name in missing_files:
                report.write(f"{name}\n")
            report.write("\n")

        if unexpected_files:
            report.write("UNEXPECTED WAV FILES\n")
            report.write("-" * 58 + "\n")
            for name in unexpected_files:
                report.write(f"{name}\n")
            report.write("\n")

        if error_counts:
            report.write("ERROR TYPES\n")
            report.write("-" * 58 + "\n")
            report.write(format_counter(error_counts))
            report.write("\n\n")

        if warning_counts:
            report.write("WARNING TYPES\n")
            report.write("-" * 58 + "\n")
            report.write(format_counter(warning_counts))
            report.write("\n\n")

        if duplicate_groups:
            report.write("DUPLICATE AUDIO GROUPS\n")
            report.write("-" * 58 + "\n")
            for digest, names in hashes.items():
                if len(names) > 1:
                    report.write(f"SHA256: {digest}\n")
                    for name in names:
                        report.write(f"  {name}\n")
            report.write("\n")

        report.write("PER-FILE RESULTS\n")
        report.write("-" * 58 + "\n")

        for result in results:
            path = result["path"]
            status = "VALID" if result["valid"] else "ERROR"

            report.write(
                f"{status} | {path.name} | "
                f"duration={result['duration']:.3f}s | "
                f"sr={result['sample_rate']} | "
                f"channels={result['channels']} | "
                f"width={result['sample_width']} | "
                f"rms={result['rms']:.6f} | "
                f"clipped={result['clipped_samples']} | "
                f"clip_ratio={result['clipping_ratio'] * 100:.4f}%\n"
            )

            if result["errors"]:
                for error in result["errors"]:
                    report.write(f"  ERROR: {error}\n")

            if result["warnings"]:
                for warning in result["warnings"]:
                    report.write(f"  WARNING: {warning}\n")

        report.write("\n")
        report.write("RESULT\n")
        report.write("-" * 58 + "\n")

        if pass_result:
            report.write("PASS\n")
            report.write(
                "English negative dataset is ready for combination "
                "with the Hindi negatives.\n"
            )
        else:
            report.write("FAIL\n")
            report.write(
                "Fix validation errors before combining datasets.\n"
            )

    print()

    if pass_result:
        print("RESULT: PASS")
        print(
            "English negative dataset is ready for combination "
            "with the Hindi negatives."
        )
        print("=" * 58)
        return 0

    print("RESULT: FAIL")
    print("Fix validation errors before continuing.")
    print("=" * 58)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
