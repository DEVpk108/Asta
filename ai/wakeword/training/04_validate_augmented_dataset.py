"""
04_validate_augmented_dataset.py

Validate the augmented ASTA wakeword dataset.

Author: ASTA
"""

from pathlib import Path
import array
import csv
import hashlib
import wave

from ai.wakeword.config import GENERATED_DIR


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

AUGMENTED_DIR = (
    GENERATED_DIR
    / "synthetic_positive"
    / "augmented"
)

MANIFEST_PATH = (
    AUGMENTED_DIR
    / "augmentation_manifest.csv"
)

REPORT_PATH = (
    AUGMENTED_DIR
    / "augmentation_validation_report.txt"
)

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2  # 16-bit PCM

MIN_DURATION_SECONDS = 0.20
MAX_DURATION_SECONDS = 4.0

SILENCE_THRESHOLD = 500

# Near full-scale threshold.
CLIPPING_THRESHOLD = 32760

# Warn only when clipping is meaningful.
CLIPPING_RATIO_WARNING = 0.001  # 0.1%

MIN_CLIPPED_SAMPLES_WARNING = 10


# ---------------------------------------------------------
# WAV analysis
# ---------------------------------------------------------

def analyze_wav(path: Path):
    """
    Analyze a mono 16-bit PCM WAV.
    """

    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()

        raw_audio = wav.readframes(
            frame_count
        )

    duration = (
        frame_count / sample_rate
        if sample_rate > 0
        else 0
    )

    peak = 0
    rms = 0
    clipped_samples = 0
    total_samples = 0

    if (
        sample_width == EXPECTED_SAMPLE_WIDTH
        and raw_audio
    ):

        samples = array.array(
            "h",
            raw_audio,
        )

        total_samples = len(samples)

        if total_samples:

            peak = max(
                abs(sample)
                for sample in samples
            )

            mean_square = sum(
                sample * sample
                for sample in samples
            ) / total_samples

            rms = mean_square ** 0.5

            clipped_samples = sum(
                1
                for sample in samples
                if abs(sample)
                >= CLIPPING_THRESHOLD
            )

    clipping_ratio = (
        clipped_samples / total_samples
        if total_samples
        else 0
    )

    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "frames": frame_count,
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "total_samples": total_samples,
        "clipped_samples": clipped_samples,
        "clipping_ratio": clipping_ratio,
    }


# ---------------------------------------------------------
# Hash
# ---------------------------------------------------------

def audio_hash(path: Path) -> str:
    """
    Calculate SHA-256 hash of a WAV file.
    """

    sha = hashlib.sha256()

    with path.open("rb") as file:

        while chunk := file.read(
            1024 * 1024
        ):

            sha.update(chunk)

    return sha.hexdigest()


# ---------------------------------------------------------
# Manifest
# ---------------------------------------------------------

def load_manifest():
    """
    Load augmentation manifest.
    """

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            "Augmentation manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    entries = {}

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "source_file",
            "gain_db",
            "noise_level",
            "speed",
            "augmentations",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        missing = (
            required_columns
            - actual_columns
        )

        if missing:

            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        for row in reader:

            filename = row["filename"]

            entries[filename] = row

    return entries


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print()
    print("=" * 50)
    print("ASTA AUGMENTED DATASET VALIDATION")
    print("=" * 50)

    if not AUGMENTED_DIR.exists():

        raise FileNotFoundError(
            "Augmented dataset directory "
            "not found:\n"
            f"{AUGMENTED_DIR}"
        )

    manifest = load_manifest()

    wav_files = sorted(
        AUGMENTED_DIR.rglob("*.wav")
    )

    print(
        f"Manifest entries : "
        f"{len(manifest)}"
    )

    print(
        f"WAV files        : "
        f"{len(wav_files)}"
    )

    print()

    errors = []
    warnings = []

    hashes = {}

    valid_count = 0

    clipping_warning_count = 0
    duplicate_count = 0
    long_clip_count = 0

    max_clipped_samples = 0
    max_clipping_ratio = 0.0

    # -----------------------------------------------------
    # Validate every WAV
    # -----------------------------------------------------

    for index, wav_path in enumerate(
        wav_files,
        start=1,
    ):

        relative = (
            wav_path
            .relative_to(
                AUGMENTED_DIR
            )
            .as_posix()
        )

        file_has_error = False

        # ---------------------------------------------
        # Read WAV
        # ---------------------------------------------

        try:

            info = analyze_wav(
                wav_path
            )

        except Exception as exc:

            errors.append(
                f"{relative}: "
                f"cannot read WAV: {exc}"
            )

            continue

        # ---------------------------------------------
        # Sample rate
        # ---------------------------------------------

        if (
            info["sample_rate"]
            != EXPECTED_SAMPLE_RATE
        ):

            errors.append(
                f"{relative}: "
                f"sample rate "
                f"{info['sample_rate']} Hz "
                f"(expected "
                f"{EXPECTED_SAMPLE_RATE} Hz)"
            )

            file_has_error = True

        # ---------------------------------------------
        # Channels
        # ---------------------------------------------

        if (
            info["channels"]
            != EXPECTED_CHANNELS
        ):

            errors.append(
                f"{relative}: "
                f"{info['channels']} channels "
                f"(expected mono)"
            )

            file_has_error = True

        # ---------------------------------------------
        # Sample width
        # ---------------------------------------------

        if (
            info["sample_width"]
            != EXPECTED_SAMPLE_WIDTH
        ):

            errors.append(
                f"{relative}: "
                f"{info['sample_width'] * 8}-bit "
                f"audio "
                f"(expected 16-bit)"
            )

            file_has_error = True

        # ---------------------------------------------
        # Duration
        # ---------------------------------------------

        if (
            info["duration"]
            < MIN_DURATION_SECONDS
        ):

            errors.append(
                f"{relative}: "
                f"too short "
                f"{info['duration']:.3f}s"
            )

            file_has_error = True

        elif (
            info["duration"]
            > MAX_DURATION_SECONDS
        ):

            warnings.append(
                f"{relative}: "
                f"long clip "
                f"{info['duration']:.3f}s"
            )

            long_clip_count += 1

        # ---------------------------------------------
        # Silence
        # ---------------------------------------------

        if (
            info["rms"]
            < SILENCE_THRESHOLD
        ):

            errors.append(
                f"{relative}: "
                f"near-silent "
                f"(RMS={info['rms']:.1f})"
            )

            file_has_error = True

        # ---------------------------------------------
        # Clipping
        # ---------------------------------------------

        clipped_samples = (
            info["clipped_samples"]
        )

        clipping_ratio = (
            info["clipping_ratio"]
        )

        max_clipped_samples = max(
            max_clipped_samples,
            clipped_samples,
        )

        max_clipping_ratio = max(
            max_clipping_ratio,
            clipping_ratio,
        )

        meaningful_clipping = (
            clipped_samples
            >= MIN_CLIPPED_SAMPLES_WARNING
            and
            clipping_ratio
            >= CLIPPING_RATIO_WARNING
        )

        if meaningful_clipping:

            warnings.append(
                f"{relative}: "
                f"possible clipping "
                f"(peak={info['peak']}, "
                f"clipped_samples="
                f"{clipped_samples}, "
                f"clipping_ratio="
                f"{clipping_ratio * 100:.4f}%)"
            )

            clipping_warning_count += 1

        # ---------------------------------------------
        # Manifest entry
        # ---------------------------------------------

        if relative not in manifest:

            errors.append(
                f"{relative}: "
                f"missing from augmentation "
                f"manifest"
            )

            file_has_error = True

        # ---------------------------------------------
        # Duplicate detection
        # ---------------------------------------------

        file_hash = audio_hash(
            wav_path
        )

        if file_hash in hashes:

            warnings.append(
                f"{relative}: "
                f"duplicate of "
                f"{hashes[file_hash]}"
            )

            duplicate_count += 1

        else:

            hashes[file_hash] = relative

        # ---------------------------------------------

        if not file_has_error:

            valid_count += 1

        if index % 100 == 0:

            print(
                f"Checked "
                f"{index}/{len(wav_files)}..."
            )

    # -----------------------------------------------------
    # Manifest → WAV consistency
    # -----------------------------------------------------

    for filename in manifest:

        path = AUGMENTED_DIR / filename

        if not path.exists():

            errors.append(
                "Manifest references "
                f"missing WAV: {filename}"
            )

    # -----------------------------------------------------
    # Source-file consistency
    # -----------------------------------------------------

    source_missing_count = 0

    for filename, row in manifest.items():

        source_file = row[
            "source_file"
        ]

        source_path = (
            GENERATED_DIR
            / "synthetic_positive"
            / "clean"
            / source_file
        )

        if not source_path.exists():

            errors.append(
                f"{filename}: "
                f"source file missing: "
                f"{source_file}"
            )

            source_missing_count += 1

    # -----------------------------------------------------
    # Manifest count consistency
    # -----------------------------------------------------

    if len(manifest) != len(wav_files):

        errors.append(
            "Manifest/WAV count mismatch: "
            f"{len(manifest)} manifest entries "
            f"vs "
            f"{len(wav_files)} WAV files"
        )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = []

    report.append(
        "ASTA AUGMENTED DATASET "
        "VALIDATION REPORT"
    )

    report.append("=" * 50)

    report.append(
        f"WAV files        : "
        f"{len(wav_files)}"
    )

    report.append(
        f"Manifest entries : "
        f"{len(manifest)}"
    )

    report.append(
        f"Valid files      : "
        f"{valid_count}"
    )

    report.append(
        f"Errors           : "
        f"{len(errors)}"
    )

    report.append(
        f"Warnings         : "
        f"{len(warnings)}"
    )

    report.append("")

    report.append(
        "VALIDATION STATISTICS"
    )

    report.append("-" * 50)

    report.append(
        f"Clipping warnings : "
        f"{clipping_warning_count}"
    )

    report.append(
        f"Duplicate warnings: "
        f"{duplicate_count}"
    )

    report.append(
        f"Long clips        : "
        f"{long_clip_count}"
    )

    report.append(
        f"Missing sources   : "
        f"{source_missing_count}"
    )

    report.append(
        f"Maximum clipped samples: "
        f"{max_clipped_samples}"
    )

    report.append(
        f"Maximum clipping ratio: "
        f"{max_clipping_ratio * 100:.4f}%"
    )

    report.append("")

    if errors:

        report.append("ERRORS")
        report.append("-" * 50)

        report.extend(errors)

        report.append("")

    if warnings:

        report.append("WARNINGS")
        report.append("-" * 50)

        report.extend(warnings)

        report.append("")

    if not errors:

        report.append(
            "RESULT: PASS"
        )

    else:

        report.append(
            "RESULT: FAIL"
        )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("AUGMENTED VALIDATION COMPLETE")
    print("=" * 50)

    print(
        f"WAV files : {len(wav_files)}"
    )

    print(
        f"Manifest  : {len(manifest)}"
    )

    print(
        f"Valid     : {valid_count}"
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
        f"{clipping_warning_count}"
    )

    print(
        f"Duplicate warnings: "
        f"{duplicate_count}"
    )

    print(
        f"Long clips        : "
        f"{long_clip_count}"
    )

    print(
        f"Missing sources   : "
        f"{source_missing_count}"
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
        f"Report    : "
        f"{REPORT_PATH}"
    )

    if not errors:

        print()
        print("RESULT: PASS")

        print(
            "Augmented dataset is ready "
            "for the next stage."
        )

    else:

        print()
        print("RESULT: FAIL")

        print(
            "Fix the reported errors "
            "before continuing."
        )

    print("=" * 50)


if __name__ == "__main__":
    main()