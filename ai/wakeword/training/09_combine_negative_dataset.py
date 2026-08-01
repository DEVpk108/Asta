"""
09_combine_negative_dataset.py

ASTA Wakeword Negative Dataset Combiner

Combines:
    - Hindi negative dataset
    - English LibriSpeech negative dataset

into:

    ai/wakeword/generated/negative/

The source manifest counts are dynamic.

The script intentionally does NOT assume:
    Hindi = 2990
    English = 3000

Instead, the manifest is treated as the source of truth.

Current expected source counts:
    Hindi   : 3287
    English : 3000
    Total   : 6287
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import wave

from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# PROJECT PATHS
# ============================================================

# File:
#   ASTA/ai/wakeword/training/09_combine_negative_dataset.py
#
# parents:
#   0 = training
#   1 = wakeword
#   2 = ai
#   3 = ASTA
#
PROJECT_ROOT = Path(__file__).resolve().parents[3]

GENERATED_DIR = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
)

DOWNLOADED_NEGATIVE_DIR = (
    GENERATED_DIR
    / "downloaded_negative"
)

# ------------------------------------------------------------
# Hindi
# ------------------------------------------------------------

HINDI_DIR = (
    DOWNLOADED_NEGATIVE_DIR
    / "hindi"
)

HINDI_MANIFEST = (
    HINDI_DIR
    / "hindi_manifest.csv"
)

# ------------------------------------------------------------
# English
# ------------------------------------------------------------

ENGLISH_DIR = DOWNLOADED_NEGATIVE_DIR

ENGLISH_MANIFEST = (
    ENGLISH_DIR
    / "librispeech_manifest.csv"
)

# ------------------------------------------------------------
# Combined output
# ------------------------------------------------------------

OUTPUT_DIR = (
    GENERATED_DIR
    / "negative"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "negative_manifest.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "negative_combination_report.txt"
)


# ============================================================
# AUDIO EXPECTATIONS
# ============================================================

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2


# ============================================================
# OUTPUT PREFIXES
# ============================================================

HINDI_PREFIX = "hindi"
ENGLISH_PREFIX = "english"


# ============================================================
# COMBINED MANIFEST SCHEMA
# ============================================================

COMBINED_FIELDNAMES = [
    "filename",
    "source",
    "source_filename",
    "text",
    "duration_seconds",
    "sample_rate",
    "channels",
    "sample_width",
    "sha256",
]


# ============================================================
# BANNER
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 58)
    print(title)
    print("=" * 58)


# ============================================================
# HASH
# ============================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 for a file.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(
    fieldnames: List[str],
    candidates: List[str],
) -> str | None:
    """
    Find a column name case-insensitively.
    """

    normalized = {
        field.strip().lower(): field
        for field in fieldnames
        if field
    }

    for candidate in candidates:
        result = normalized.get(
            candidate.lower()
        )

        if result:
            return result

    return None


def get_value(
    row: Dict[str, str],
    candidates: List[str],
    default: str = "",
) -> str:
    """
    Get the first non-empty value from a row.
    """

    normalized = {
        str(key).strip().lower(): value
        for key, value in row.items()
        if key
    }

    for candidate in candidates:
        value = normalized.get(
            candidate.lower()
        )

        if value is not None:
            value = str(value).strip()

            if value:
                return value

    return default


def get_float(
    row: Dict[str, str],
    candidates: List[str],
) -> float | None:
    """
    Get a floating-point value from a manifest row.
    """

    value = get_value(
        row,
        candidates,
        "",
    )

    if not value:
        return None

    try:
        return float(value)

    except ValueError:
        return None


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest(
    manifest_path: Path,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Load a CSV manifest.

    Returns:
        rows
        filename column name
    """

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Manifest missing:\n"
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise RuntimeError(
                "Manifest has no header:\n"
                f"{manifest_path}"
            )

        fieldnames = list(
            reader.fieldnames
        )

        filename_column = find_column(
            fieldnames,
            [
                "filename",
                "file",
                "wav",
                "wav_filename",
                "output_filename",
            ],
        )

        if filename_column is None:
            raise RuntimeError(
                "Could not find a filename column in:\n"
                f"{manifest_path}\n\n"
                f"Columns found:\n"
                f"{fieldnames}"
            )

        rows: List[Dict[str, str]] = []

        for row in reader:
            if not row:
                continue

            filename = (
                row.get(
                    filename_column,
                    "",
                )
                or ""
            ).strip()

            if not filename:
                continue

            rows.append(row)

    return rows, filename_column


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_source_file(
    source_dir: Path,
    filename: str,
) -> Path:
    """
    Resolve a manifest filename robustly.

    Supported forms:

    1. Bare filename:
       hindi_00000.wav

    2. Source-relative:
       hindi/hindi_00000.wav

    3. Project-relative:
       ai/wakeword/generated/downloaded_negative/hindi/hindi_00000.wav

    4. Absolute project path:
       E:/Projects/ASTA/ai/wakeword/...

    The resolved file must remain inside PROJECT_ROOT.
    """

    raw = str(filename).strip()

    if not raw:
        raise RuntimeError(
            "Manifest contains an empty filename."
        )

    # --------------------------------------------------------
    # Normalize separators.
    # --------------------------------------------------------

    normalized = raw.replace(
        "\\",
        "/",
    )

    candidate = Path(normalized)

    project_root_resolved = (
        PROJECT_ROOT.resolve()
    )

    source_dir_resolved = (
        source_dir.resolve()
    )

    # --------------------------------------------------------
    # Absolute path.
    # --------------------------------------------------------

    if candidate.is_absolute():

        resolved = candidate.resolve()

        try:
            resolved.relative_to(
                project_root_resolved
            )

        except ValueError as exc:
            raise RuntimeError(
                "Manifest contains an absolute path "
                "outside the project:\n"
                f"{raw}"
            ) from exc

        return resolved

    # --------------------------------------------------------
    # Project-relative path.
    #
    # Example:
    #
    # ai/wakeword/generated/
    # downloaded_negative/hindi/hindi_00000.wav
    # --------------------------------------------------------

    project_candidate = (
        project_root_resolved
        / candidate
    ).resolve()

    try:
        project_candidate.relative_to(
            project_root_resolved
        )

    except ValueError as exc:
        raise RuntimeError(
            "Manifest contains a path outside "
            "the project:\n"
            f"{raw}"
        ) from exc

    if project_candidate.is_file():
        return project_candidate

    # --------------------------------------------------------
    # Source-relative path.
    #
    # Example:
    #
    # source_dir / hindi_00000.wav
    # --------------------------------------------------------

    source_candidate = (
        source_dir_resolved
        / candidate
    ).resolve()

    try:
        source_candidate.relative_to(
            project_root_resolved
        )

    except ValueError as exc:
        raise RuntimeError(
            "Manifest path escapes project root:\n"
            f"{raw}"
        ) from exc

    if source_candidate.is_file():
        return source_candidate

    # --------------------------------------------------------
    # Basename fallback.
    #
    # Useful when manifest contains:
    #
    # some/prefix/hindi_00000.wav
    #
    # but actual file is directly inside source_dir.
    # --------------------------------------------------------

    basename_candidate = (
        source_dir_resolved
        / candidate.name
    ).resolve()

    try:
        basename_candidate.relative_to(
            project_root_resolved
        )

    except ValueError as exc:
        raise RuntimeError(
            "Manifest basename escapes project root:\n"
            f"{raw}"
        ) from exc

    if basename_candidate.is_file():
        return basename_candidate

    # --------------------------------------------------------
    # Return the most useful failed candidate.
    # --------------------------------------------------------

    return project_candidate


# ============================================================
# WAV VALIDATION
# ============================================================

def validate_wav(
    path: Path,
) -> None:
    """
    Validate basic WAV properties.
    """

    if not path.is_file():
        raise RuntimeError(
            "WAV file does not exist:\n"
            f"{path}"
        )

    try:
        with wave.open(
            str(path),
            "rb",
        ) as wav:

            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()

    except Exception as exc:
        raise RuntimeError(
            "Could not read WAV:\n"
            f"{path}\n"
            f"Error: {exc}"
        ) from exc

    if channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Unexpected channel count:\n"
            f"File     : {path}\n"
            f"Expected : {EXPECTED_CHANNELS}\n"
            f"Actual   : {channels}"
        )

    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise RuntimeError(
            "Unexpected sample width:\n"
            f"File     : {path}\n"
            f"Expected : {EXPECTED_SAMPLE_WIDTH}\n"
            f"Actual   : {sample_width}"
        )

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise RuntimeError(
            "Unexpected sample rate:\n"
            f"File     : {path}\n"
            f"Expected : {EXPECTED_SAMPLE_RATE}\n"
            f"Actual   : {sample_rate}"
        )

    if frame_count <= 0:
        raise RuntimeError(
            "WAV contains no audio frames:\n"
            f"{path}"
        )


# ============================================================
# SOURCE MANIFEST VALIDATION
# ============================================================

def validate_source_manifest(
    name: str,
    source_dir: Path,
    manifest_path: Path,
) -> Tuple[
    List[Dict[str, str]],
    str,
]:
    """
    Validate a source manifest dynamically.

    There is intentionally no hard-coded sample count.
    """

    print()
    print(
        f"Validating {name} manifest..."
    )

    print(
        f"Manifest : {manifest_path}"
    )

    rows, filename_column = (
        load_manifest(
            manifest_path
        )
    )

    print(
        f"Manifest entries : {len(rows)}"
    )

    if not rows:
        raise RuntimeError(
            f"{name} manifest contains no entries:\n"
            f"{manifest_path}"
        )

    # --------------------------------------------------------
    # Extract filenames.
    # --------------------------------------------------------

    filenames = [
        (
            row.get(
                filename_column,
                "",
            )
            or ""
        ).strip()
        for row in rows
    ]

    # --------------------------------------------------------
    # Duplicate manifest references.
    # --------------------------------------------------------

    seen = set()
    duplicate_names = []

    for filename in filenames:

        if filename in seen:
            duplicate_names.append(
                filename
            )

        seen.add(filename)

    if duplicate_names:

        preview = "\n".join(
            f"  - {item}"
            for item in duplicate_names[:20]
        )

        raise RuntimeError(
            f"{name} manifest contains "
            f"{len(duplicate_names)} duplicate "
            "filename references.\n\n"
            f"{preview}"
        )

    # --------------------------------------------------------
    # Resolve and verify every referenced WAV.
    # --------------------------------------------------------

    missing: List[str] = []

    resolved_files: List[Path] = []

    for filename in filenames:

        path = resolve_source_file(
            source_dir,
            filename,
        )

        if not path.is_file():
            missing.append(filename)

        else:
            resolved_files.append(path)

    if missing:

        preview = "\n".join(
            f"  - {item}"
            for item in missing[:20]
        )

        raise RuntimeError(
            f"{name} manifest references "
            f"{len(missing)} missing WAV files.\n\n"
            f"{preview}"
        )

    # --------------------------------------------------------
    # Determine actual WAV files directly in source directory.
    # --------------------------------------------------------

    actual_wavs = {
        path.resolve()
        for path in source_dir.glob(
            "*.wav"
        )
        if path.is_file()
    }

    resolved_set = set(
        resolved_files
    )

    unexpected = sorted(
        actual_wavs - resolved_set
    )

    if unexpected:
        print(
            f"Warning: {len(unexpected)} WAV files "
            f"exist outside the {name} manifest."
        )

        print(
            "These files will NOT be included."
        )

    print(
        f"{name} manifest validation : OK"
    )

    return (
        rows,
        filename_column,
    )


# ============================================================
# DESTINATION NAME
# ============================================================

def make_destination_name(
    prefix: str,
    index: int,
) -> str:
    """
    Generate a deterministic combined filename.
    """

    return (
        f"{prefix}_{index:06d}.wav"
    )


# ============================================================
# COMBINED MANIFEST ROW
# ============================================================

def build_combined_row(
    *,
    destination_filename: str,
    source_name: str,
    source_filename: str,
    source_row: Dict[str, str],
    wav_path: Path,
) -> Dict[str, str]:

    duration = get_float(
        source_row,
        [
            "duration_seconds",
            "duration",
            "source_duration_seconds",
        ],
    )

    text = get_value(
        source_row,
        [
            "text",
            "sentence",
            "transcript",
            "transcription",
        ],
        "",
    )

    with wave.open(
        str(wav_path),
        "rb",
    ) as wav:

        sample_rate = (
            wav.getframerate()
        )

        channels = (
            wav.getnchannels()
        )

        sample_width = (
            wav.getsampwidth()
        )

        frame_count = (
            wav.getnframes()
        )

    if duration is None:

        if sample_rate > 0:
            duration = (
                frame_count
                / sample_rate
            )

        else:
            duration = 0.0

    return {
        "filename": destination_filename,
        "source": source_name,
        "source_filename": source_filename,
        "text": text,
        "duration_seconds": (
            f"{duration:.6f}"
        ),
        "sample_rate": str(
            sample_rate
        ),
        "channels": str(
            channels
        ),
        "sample_width": str(
            sample_width
        ),
        "sha256": sha256_file(
            wav_path
        ),
    }


# ============================================================
# PREPARE OUTPUT
# ============================================================

def prepare_output_directory() -> None:
    """
    Remove the previous combined dataset and recreate it.

    IMPORTANT:
        Only generated/negative is removed.

        downloaded_negative is untouched.
    """

    if OUTPUT_DIR.exists():

        print()
        print(
            "Removing previous combined dataset..."
        )

        print(
            f"Output : {OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# COPY SOURCE DATASET
# ============================================================

def copy_source_dataset(
    *,
    source_name: str,
    source_dir: Path,
    rows: List[Dict[str, str]],
    filename_column: str,
    prefix: str,
) -> List[Dict[str, str]]:
    """
    Copy one source dataset into the combined dataset.

    Each source gets its own filename prefix, preventing collisions.
    """

    print()
    print(
        f"Combining {source_name} negatives..."
    )

    combined_rows: List[
        Dict[str, str]
    ] = []

    total = len(rows)

    for position, row in enumerate(
        rows,
        start=1,
    ):

        source_filename = (
            row.get(
                filename_column,
                "",
            )
            or ""
        ).strip()

        if not source_filename:
            raise RuntimeError(
                f"{source_name} manifest contains "
                "an empty filename."
            )

        source_path = (
            resolve_source_file(
                source_dir,
                source_filename,
            )
        )

        if not source_path.is_file():
            raise RuntimeError(
                "Source WAV disappeared during "
                "combination:\n"
                f"{source_path}"
            )

        # ----------------------------------------------------
        # Validate original source WAV.
        # ----------------------------------------------------

        validate_wav(
            source_path
        )

        # ----------------------------------------------------
        # Destination index.
        #
        # Each source starts at zero because prefixes differ.
        # ----------------------------------------------------

        destination_filename = (
            make_destination_name(
                prefix,
                position - 1,
            )
        )

        destination_path = (
            OUTPUT_DIR
            / destination_filename
        )

        if destination_path.exists():
            raise RuntimeError(
                "Destination filename collision:\n"
                f"{destination_path}"
            )

        # ----------------------------------------------------
        # Copy.
        # ----------------------------------------------------

        shutil.copy2(
            source_path,
            destination_path,
        )

        # ----------------------------------------------------
        # Build unified manifest row.
        # ----------------------------------------------------

        combined_row = (
            build_combined_row(
                destination_filename=(
                    destination_filename
                ),
                source_name=source_name,
                source_filename=(
                    source_filename
                ),
                source_row=row,
                wav_path=destination_path,
            )
        )

        combined_rows.append(
            combined_row
        )

        if (
            position % 100 == 0
            or position == total
        ):
            print(
                f"  {source_name}: "
                f"{position}/{total}"
            )

    print(
        f"{source_name} combined : "
        f"{len(combined_rows)}"
    )

    return combined_rows


# ============================================================
# WRITE COMBINED MANIFEST
# ============================================================

def write_manifest(
    rows: List[Dict[str, str]],
) -> None:

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=COMBINED_FIELDNAMES,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

def validate_combined_dataset(
    rows: List[Dict[str, str]],
) -> None:

    print()
    print(
        "Validating combined dataset..."
    )

    # --------------------------------------------------------
    # Manifest filenames.
    # --------------------------------------------------------

    manifest_names = [
        row["filename"]
        for row in rows
    ]

    manifest_set = set(
        manifest_names
    )

    # --------------------------------------------------------
    # Duplicate destination filenames.
    # --------------------------------------------------------

    if len(manifest_names) != len(
        manifest_set
    ):
        raise RuntimeError(
            "Combined manifest contains "
            "duplicate filenames."
        )

    # --------------------------------------------------------
    # Actual WAV files.
    # --------------------------------------------------------

    actual_names = {
        path.name
        for path in OUTPUT_DIR.glob(
            "*.wav"
        )
        if path.is_file()
    }

    # --------------------------------------------------------
    # Missing.
    # --------------------------------------------------------

    missing = sorted(
        manifest_set - actual_names
    )

    # --------------------------------------------------------
    # Unexpected.
    # --------------------------------------------------------

    unexpected = sorted(
        actual_names - manifest_set
    )

    if missing:

        preview = "\n".join(
            f"  - {item}"
            for item in missing[:20]
        )

        raise RuntimeError(
            f"Combined dataset is missing "
            f"{len(missing)} WAV files.\n\n"
            f"{preview}"
        )

    if unexpected:

        preview = "\n".join(
            f"  - {item}"
            for item in unexpected[:20]
        )

        raise RuntimeError(
            f"Combined dataset contains "
            f"{len(unexpected)} unexpected WAV files.\n\n"
            f"{preview}"
        )

    # --------------------------------------------------------
    # Validate every combined WAV.
    # --------------------------------------------------------

    total = len(
        manifest_names
    )

    for index, filename in enumerate(
        manifest_names,
        start=1,
    ):

        path = (
            OUTPUT_DIR
            / filename
        )

        validate_wav(
            path
        )

        if (
            index % 500 == 0
            or index == total
        ):
            print(
                f"  Final WAV validation: "
                f"{index}/{total}"
            )

    print(
        "Combined filesystem/manifest mapping : OK"
    )


# ============================================================
# REPORT
# ============================================================

def write_report(
    *,
    hindi_count: int,
    english_count: int,
    total_count: int,
) -> None:

    report_lines = [
        "ASTA NEGATIVE DATASET COMBINATION REPORT",
        "=" * 58,
        "",
        f"Hindi negatives   : {hindi_count}",
        f"English negatives : {english_count}",
        f"Combined negatives: {total_count}",
        "",
        f"Hindi manifest   : {HINDI_MANIFEST}",
        f"English manifest : {ENGLISH_MANIFEST}",
        f"Output directory : {OUTPUT_DIR}",
        f"Output manifest  : {OUTPUT_MANIFEST}",
        "",
        "RESULT: PASS",
        "",
    ]

    REPORT_FILE.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    banner(
        "ASTA NEGATIVE DATASET COMBINATION"
    )

    print(
        f"Hindi manifest   : {HINDI_MANIFEST}"
    )

    print(
        f"English manifest : {ENGLISH_MANIFEST}"
    )

    print(
        f"Output           : {OUTPUT_DIR}"
    )

    print(
        f"Output manifest  : {OUTPUT_MANIFEST}"
    )

    # ========================================================
    # PATH CHECK
    # ========================================================

    print()
    print(
        "Path check..."
    )

    print(
        "Hindi manifest exists   : "
        f"{HINDI_MANIFEST.is_file()}"
    )

    print(
        "English manifest exists : "
        f"{ENGLISH_MANIFEST.is_file()}"
    )

    if not HINDI_MANIFEST.is_file():
        raise FileNotFoundError(
            "Hindi manifest missing:\n"
            f"{HINDI_MANIFEST}"
        )

    if not ENGLISH_MANIFEST.is_file():
        raise FileNotFoundError(
            "English manifest missing:\n"
            f"{ENGLISH_MANIFEST}"
        )

    # ========================================================
    # HINDI
    # ========================================================

    hindi_rows, hindi_filename_column = (
        validate_source_manifest(
            name="Hindi",
            source_dir=HINDI_DIR,
            manifest_path=HINDI_MANIFEST,
        )
    )

    # ========================================================
    # ENGLISH
    # ========================================================

    english_rows, english_filename_column = (
        validate_source_manifest(
            name="English",
            source_dir=ENGLISH_DIR,
            manifest_path=ENGLISH_MANIFEST,
        )
    )

    # ========================================================
    # DYNAMIC COUNTS
    # ========================================================

    hindi_count = len(
        hindi_rows
    )

    english_count = len(
        english_rows
    )

    total_count = (
        hindi_count
        + english_count
    )

    print()
    print(
        "Dynamic source counts:"
    )

    print(
        f"  Hindi   : {hindi_count}"
    )

    print(
        f"  English : {english_count}"
    )

    print(
        f"  Total   : {total_count}"
    )

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    prepare_output_directory()

    # ========================================================
    # COPY HINDI
    # ========================================================

    combined_hindi = (
        copy_source_dataset(
            source_name="Hindi",
            source_dir=HINDI_DIR,
            rows=hindi_rows,
            filename_column=(
                hindi_filename_column
            ),
            prefix=HINDI_PREFIX,
        )
    )

    # ========================================================
    # COPY ENGLISH
    # ========================================================

    combined_english = (
        copy_source_dataset(
            source_name="English",
            source_dir=ENGLISH_DIR,
            rows=english_rows,
            filename_column=(
                english_filename_column
            ),
            prefix=ENGLISH_PREFIX,
        )
    )

    # ========================================================
    # COMBINED ROWS
    # ========================================================

    combined_rows = (
        combined_hindi
        + combined_english
    )

    final_count = len(
        combined_rows
    )

    if final_count != total_count:
        raise RuntimeError(
            "Combined count mismatch.\n"
            f"Expected : {total_count}\n"
            f"Actual   : {final_count}"
        )

    # ========================================================
    # WRITE MANIFEST
    # ========================================================

    print()
    print(
        "Writing combined manifest..."
    )

    write_manifest(
        combined_rows
    )

    print(
        f"Manifest written : "
        f"{OUTPUT_MANIFEST}"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validate_combined_dataset(
        combined_rows
    )

    # ========================================================
    # REPORT
    # ========================================================

    write_report(
        hindi_count=hindi_count,
        english_count=english_count,
        total_count=final_count,
    )

    # ========================================================
    # FINAL
    # ========================================================

    banner(
        "NEGATIVE DATASET COMBINATION COMPLETE"
    )

    print(
        f"Hindi negatives   : {hindi_count}"
    )

    print(
        f"English negatives : {english_count}"
    )

    print(
        f"Combined negatives: {final_count}"
    )

    print(
        f"Manifest          : "
        f"{OUTPUT_MANIFEST}"
    )

    print(
        f"Report            : "
        f"{REPORT_FILE}"
    )

    print(
        "=" * 58
    )

    print()
    print(
        "RESULT: PASS"
    )

    print(
        "Hindi and English negative datasets "
        "were combined successfully."
    )

    print(
        "=" * 58
    )


if __name__ == "__main__":
    main()