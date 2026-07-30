"""
02_validate_dataset.py

Validate the clean synthetic ASTA wakeword dataset.

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

DATASET_DIR = (
    GENERATED_DIR
    / "synthetic_positive"
    / "clean"
)

MANIFEST_PATH = DATASET_DIR / "manifest.csv"

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2  # 16-bit PCM

MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = 3.0

# Near-full-scale sample threshold.
# 32767 is maximum positive 16-bit PCM.
CLIPPING_THRESHOLD = 32760

# Only warn when a meaningful percentage of samples
# are actually at/near full scale.
CLIPPING_RATIO_WARNING = 0.001  # 0.1%

# Also warn if there are a large number of clipped samples,
# even if the percentage is small.
MIN_CLIPPED_SAMPLES_WARNING = 10

# Silence threshold for RMS.
SILENCE_THRESHOLD = 500

REPORT_PATH = (
    DATASET_DIR
    / "validation_report.txt"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def audio_hash(path: Path) -> str:
    """
    Return SHA-256 hash of the complete WAV file.
    """

    sha = hashlib.sha256()

    with path.open("rb") as file:

        while chunk := file.read(
            1024 * 1024
        ):

            sha.update(chunk)

    return sha.hexdigest()


# ---------------------------------------------------------

def analyze_wav(path: Path):
    """
    Analyze one WAV file.

    Returns audio properties including:
        - sample rate
        - channels
        - duration
        - RMS
        - peak
        - clipped sample count
        - clipping ratio
    """

    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()

        duration = (
            frame_count / sample_rate
            if sample_rate > 0
            else 0
        )

        raw_audio = wav.readframes(
            frame_count
        )

    peak = 0
    rms = 0
    clipped_samples = 0
    total_samples = 0

    # -----------------------------------------------------
    # 16-bit PCM analysis
    # -----------------------------------------------------

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

            # Count samples near either positive
            # or negative full-scale.
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
# Manifest
# ---------------------------------------------------------

def load_manifest():
    """
    Load manifest entries into a dictionary.
    """

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            "Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    entries = {}

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            filename = row["filename"]

            entries[filename] = row

    return entries


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def main():

    print()
    print("=" * 50)
    print("ASTA DATASET VALIDATION")
    print("=" * 50)

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            "Dataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    manifest = load_manifest()

    wav_files = sorted(
        DATASET_DIR.rglob("*.wav")
    )

    print(
        f"Manifest entries : {len(manifest)}"
    )

    print(
        f"WAV files        : {len(wav_files)}"
    )

    print()

    errors = []
    warnings = []

    hashes = {}

    valid_count = 0

    # Statistics
    clipping_warning_count = 0
    duplicate_count = 0
    long_clip_count = 0

    max_clipped_samples = 0
    max_clipping_ratio = 0.0

    # -----------------------------------------------------
    # Validate WAV files
    # -----------------------------------------------------

    for index, wav_path in enumerate(
        wav_files,
        start=1,
    ):

        relative = (
            wav_path
            .relative_to(DATASET_DIR)
            .as_posix()
        )

        file_has_error = False

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

        # -------------------------------------------------
        # Sample rate
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Channels
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Sample width
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Duration
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Silence
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Clipping
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Manifest
        # -------------------------------------------------

        if relative not in manifest:

            errors.append(
                f"{relative}: "
                f"missing from manifest"
            )

            file_has_error = True

        # -------------------------------------------------
        # Duplicate detection
        # -------------------------------------------------

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

        # -------------------------------------------------

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

        path = DATASET_DIR / filename

        if not path.exists():

            errors.append(
                "Manifest references "
                f"missing WAV: {filename}"
            )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = []

    report.append(
        "ASTA DATASET VALIDATION REPORT"
    )

    report.append("=" * 50)

    report.append(
        f"WAV files        : {len(wav_files)}"
    )

    report.append(
        f"Manifest entries : {len(manifest)}"
    )

    report.append(
        f"Valid files      : {valid_count}"
    )

    report.append(
        f"Errors           : {len(errors)}"
    )

    report.append(
        f"Warnings         : {len(warnings)}"
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
        f"Maximum clipped samples "
        f"in one file: "
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
    print("VALIDATION COMPLETE")
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
            "Dataset is structurally ready "
            "for further inspection."
        )

    else:

        print()
        print("RESULT: FAIL")

        print(
            "Fix the reported errors "
            "before augmentation."
        )

    print("=" * 50)


if __name__ == "__main__":
    main()