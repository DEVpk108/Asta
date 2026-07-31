"""
07_validate_hindi_negative_dataset.py

Validate the complete Hindi negative wake-word dataset.

Expected manifest format:

filename,source_file,text,duration,source_sample_rate,sample_rate,sha256

Example:

hindi_000000.wav,common_voice_hi_27388723.mp3,हिंदी text,4.428000,32000,22050,<sha256>

Author: ASTA
"""

from pathlib import Path
import csv
import hashlib
import wave
from collections import Counter

from ai.wakeword.config import GENERATED_DIR


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = (
    GENERATED_DIR
    / "downloaded_negative"
    / "hindi"
)

MANIFEST_PATH = DATASET_DIR / "hindi_manifest.csv"

REPORT_PATH = (
    DATASET_DIR
    / "hindi_validation_report.txt"
)

EXPECTED_SAMPLE_RATE = 22050

MIN_DURATION = 0.25
MAX_DURATION = 10.0

CLIP_THRESHOLD = 32767
CLIPPING_RATIO_WARNING = 0.001  # 0.1%

CHECKPOINT = 100


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_filename(value: str) -> str:

    """
    Normalize filenames for reliable comparison.

    Manifest:
        hindi_000000.wav

    Filesystem:
        E:\\...\\hindi\\hindi_000000.wav
    """

    value = str(value).strip()

    value = value.replace("\\", "/")

    return Path(value).name


def read_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    rows = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.reader(f)

        for line_number, row in enumerate(
            reader,
            start=1,
        ):

            if not row:
                continue

            # Skip header if present.
            first = row[0].strip().lower()

            if first in {
                "filename",
                "file",
                "wav",
                "wav_file",
            }:

                continue

            if len(row) < 1:
                continue

            filename = normalize_filename(row[0])

            if not filename:
                continue

            rows.append(
                {
                    "line": line_number,
                    "filename": filename,
                    "row": row,
                }
            )

    return rows


def find_wav_files():

    """
    Recursively find all WAV files.

    Return filenames relative to DATASET_DIR.
    """

    files = {}

    for path in DATASET_DIR.rglob("*.wav"):

        relative = path.relative_to(
            DATASET_DIR
        )

        relative_name = normalize_filename(
            relative.as_posix()
        )

        files[relative_name] = path

    return files


def inspect_wav(path: Path):

    errors = []
    warnings = []

    try:

        with wave.open(
            str(path),
            "rb",
        ) as wav:

            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()

            if sample_rate != EXPECTED_SAMPLE_RATE:

                errors.append(
                    f"sample_rate={sample_rate}"
                )

            if channels != 1:

                errors.append(
                    f"channels={channels}"
                )

            if sample_width != 2:

                errors.append(
                    f"sample_width={sample_width * 8}bit"
                )

            if frame_count <= 0:

                errors.append(
                    "empty_audio"
                )

            duration = (
                frame_count / sample_rate
                if sample_rate > 0
                else 0
            )

            if duration < MIN_DURATION:

                warnings.append(
                    f"short_clip={duration:.3f}s"
                )

            if duration > MAX_DURATION:

                warnings.append(
                    f"long_clip={duration:.3f}s"
                )

            # ------------------------------------------------
            # Check PCM clipping.
            # ------------------------------------------------

            wav.rewind()

            raw = wav.readframes(
                frame_count
            )

            clipped_samples = 0
            total_samples = 0

            if sample_width == 2:

                for i in range(
                    0,
                    len(raw) - 1,
                    2,
                ):

                    value = int.from_bytes(
                        raw[i:i + 2],
                        byteorder="little",
                        signed=True,
                    )

                    total_samples += 1

                    if abs(value) >= CLIP_THRESHOLD:

                        clipped_samples += 1

            clipping_ratio = (
                clipped_samples / total_samples
                if total_samples
                else 0
            )

            if (
                clipping_ratio
                >= CLIPPING_RATIO_WARNING
            ):

                warnings.append(
                    "clipping_ratio="
                    f"{clipping_ratio * 100:.4f}%"
                )

            return {
                "errors": errors,
                "warnings": warnings,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "clipped_samples": clipped_samples,
                "clipping_ratio": clipping_ratio,
            }

    except Exception as exc:

        return {
            "errors": [
                f"wav_read_error={exc}"
            ],
            "warnings": [],
            "duration": 0,
            "sample_rate": 0,
            "channels": 0,
            "sample_width": 0,
            "clipped_samples": 0,
            "clipping_ratio": 0,
        }


# ============================================================
# VALIDATION
# ============================================================

def main():

    print()
    print("=" * 50)
    print("ASTA HINDI NEGATIVE DATASET VALIDATION")
    print("=" * 50)

    print(
        f"Dataset : {DATASET_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print()

    if not DATASET_DIR.exists():

        print(
            "ERROR: Dataset directory does not exist."
        )

        return 1

    if not MANIFEST_PATH.exists():

        print(
            "ERROR: Manifest does not exist."
        )

        return 1

    # --------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------

    manifest_rows = read_manifest()

    manifest_names = [
        row["filename"]
        for row in manifest_rows
    ]

    manifest_set = set(
        manifest_names
    )

    # --------------------------------------------------------
    # Find WAVs
    # --------------------------------------------------------

    wav_files = find_wav_files()

    wav_set = set(
        wav_files.keys()
    )

    print(
        f"Manifest entries : {len(manifest_rows)}"
    )

    print(
        f"WAV files        : {len(wav_files)}"
    )

    print()

    # --------------------------------------------------------
    # Detect duplicates in manifest
    # --------------------------------------------------------

    manifest_counter = Counter(
        manifest_names
    )

    duplicate_manifest = {
        name: count
        for name, count
        in manifest_counter.items()
        if count > 1
    }

    # --------------------------------------------------------
    # Correct filename comparison
    # --------------------------------------------------------

    missing = sorted(
        manifest_set - wav_set
    )

    unexpected = sorted(
        wav_set - manifest_set
    )

    if missing:

        print(
            f"Missing WAV files : {len(missing)}"
        )

    if unexpected:

        print(
            f"Unexpected WAVs   : {len(unexpected)}"
        )

    if not missing and not unexpected:

        print(
            "Filesystem/manifest mapping : OK"
        )

    print()

    # --------------------------------------------------------
    # Validation counters
    # --------------------------------------------------------

    valid = 0
    errors = []
    warnings = []

    clipping_warnings = 0
    duplicate_warnings = len(
        duplicate_manifest
    )

    silent_clips = 0
    duration_warnings = 0
    format_warnings = 0

    max_clipped_samples = 0
    max_clipping_ratio = 0.0

    # --------------------------------------------------------
    # Missing/unexpected errors
    # --------------------------------------------------------

    for filename in missing:

        errors.append(
            f"MISSING: {filename}"
        )

    for filename in unexpected:

        errors.append(
            f"UNEXPECTED: {filename}"
        )

    # --------------------------------------------------------
    # Validate manifest WAVs
    # --------------------------------------------------------

    total = len(manifest_rows)

    for index, entry in enumerate(
        manifest_rows,
        start=1,
    ):

        filename = entry["filename"]

        path = wav_files.get(
            filename
        )

        if path is None:

            if (
                index % CHECKPOINT == 0
                or index == total
            ):

                print(
                    f"Checked {index}/{total}..."
                )

            continue

        result = inspect_wav(
            path
        )

        file_errors = result[
            "errors"
        ]

        file_warnings = result[
            "warnings"
        ]

        max_clipped_samples = max(
            max_clipped_samples,
            result["clipped_samples"],
        )

        max_clipping_ratio = max(
            max_clipping_ratio,
            result["clipping_ratio"],
        )

        if file_errors:

            for error in file_errors:

                errors.append(
                    f"{filename}: {error}"
                )

                if (
                    "sample_rate"
                    in error
                    or "channels"
                    in error
                    or "sample_width"
                    in error
                ):

                    format_warnings += 1

        if file_warnings:

            for warning in file_warnings:

                warnings.append(
                    f"{filename}: {warning}"
                )

                if (
                    "clipping"
                    in warning
                ):

                    clipping_warnings += 1

                if (
                    "long_clip"
                    in warning
                    or "short_clip"
                    in warning
                ):

                    duration_warnings += 1

        # ----------------------------------------------------
        # Silent clip check
        # ----------------------------------------------------

        try:

            with wave.open(
                str(path),
                "rb",
            ) as wav:

                frames = wav.readframes(
                    wav.getnframes()
                )

                if not any(frames):

                    silent_clips += 1

                    errors.append(
                        f"{filename}: silent_audio"
                    )

        except Exception:
            pass

        if not file_errors:

            valid += 1

        if (
            index % CHECKPOINT == 0
            or index == total
        ):

            print(
                f"Checked {index}/{total}..."
            )

    # --------------------------------------------------------
    # Duplicate files by SHA256
    # --------------------------------------------------------

    hashes = {}

    duplicate_files = []

    for filename, path in wav_files.items():

        try:

            digest = sha256_file(
                path
            )

        except Exception as exc:

            errors.append(
                f"{filename}: "
                f"hash_error={exc}"
            )

            continue

        if digest in hashes:

            duplicate_files.append(
                (
                    filename,
                    hashes[digest],
                )
            )

        else:

            hashes[digest] = filename

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Duplicate WAV contents are warnings, not structural
    # failures. This is especially useful for negative
    # datasets where duplicate source material can occur.
    # --------------------------------------------------------

    if duplicate_files:

        duplicate_warnings += len(
            duplicate_files
        )

        for current, original in duplicate_files:

            warnings.append(
                "DUPLICATE: "
                f"{current} == {original}"
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "ASTA HINDI NEGATIVE DATASET VALIDATION"
    )

    report_lines.append(
        "=" * 50
    )

    report_lines.append(
        f"Dataset : {DATASET_DIR}"
    )

    report_lines.append(
        f"Manifest: {MANIFEST_PATH}"
    )

    report_lines.append("")

    report_lines.append(
        f"Manifest entries : {len(manifest_rows)}"
    )

    report_lines.append(
        f"WAV files        : {len(wav_files)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Valid             : {valid}"
    )

    report_lines.append(
        f"Errors            : {len(errors)}"
    )

    report_lines.append(
        f"Warnings          : {len(warnings)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Missing files     : {len(missing)}"
    )

    report_lines.append(
        f"Unexpected files  : {len(unexpected)}"
    )

    report_lines.append(
        f"Clipping warnings : {clipping_warnings}"
    )

    report_lines.append(
        f"Duplicate warnings: {duplicate_warnings}"
    )

    report_lines.append(
        f"Silent clips      : {silent_clips}"
    )

    report_lines.append(
        f"Duration warnings : {duration_warnings}"
    )

    report_lines.append(
        f"Format warnings   : {format_warnings}"
    )

    report_lines.append(
        f"Max clipped samples: "
        f"{max_clipped_samples}"
    )

    report_lines.append(
        f"Max clipping ratio: "
        f"{max_clipping_ratio * 100:.4f}%"
    )

    report_lines.append("")

    if errors:

        report_lines.append(
            "ERRORS"
        )

        report_lines.append(
            "-" * 50
        )

        report_lines.extend(
            errors
        )

        report_lines.append("")

    if warnings:

        report_lines.append(
            "WARNINGS"
        )

        report_lines.append(
            "-" * 50
        )

        report_lines.extend(
            warnings
        )

        report_lines.append("")

    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    structural_failure = (
        len(missing) > 0
        or len(unexpected) > 0
        or len(errors) > 0
    )

    if structural_failure:

        result_text = "FAIL"

        report_lines.append(
            "RESULT: FAIL"
        )

        report_lines.append(
            "Fix validation errors before continuing."
        )

    else:

        result_text = "PASS"

        report_lines.append(
            "RESULT: PASS"
        )

        report_lines.append(
            "Hindi negative dataset is ready "
            "for combination with other negatives."
        )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "HINDI NEGATIVE VALIDATION COMPLETE"
    )
    print("=" * 50)

    print(
        f"WAV files : {len(wav_files)}"
    )

    print(
        f"Manifest  : {len(manifest_rows)}"
    )

    print(
        f"Valid     : {valid}"
    )

    print(
        f"Errors    : {len(errors)}"
    )

    print(
        f"Warnings  : {len(warnings)}"
    )

    print()

    print(
        f"Clipping warnings : "
        f"{clipping_warnings}"
    )

    print(
        f"Duplicate warnings: "
        f"{duplicate_warnings}"
    )

    print(
        f"Silent clips      : "
        f"{silent_clips}"
    )

    print(
        f"Duration warnings : "
        f"{duration_warnings}"
    )

    print(
        f"Format warnings   : "
        f"{format_warnings}"
    )

    print(
        f"Missing files     : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected files  : "
        f"{len(unexpected)}"
    )

    print(
        f"Max clipped samples: "
        f"{max_clipped_samples}"
    )

    print(
        f"Max clipping ratio: "
        f"{max_clipping_ratio * 100:.4f}%"
    )

    print()

    print(
        f"Report    : {REPORT_PATH}"
    )

    print()

    print(
        f"RESULT: {result_text}"
    )

    if result_text == "PASS":

        print(
            "Hindi negative dataset is ready "
            "for combination with other negatives."
        )

    else:

        print(
            "Fix validation errors before continuing."
        )

    print("=" * 50)

    return 0 if result_text == "PASS" else 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )