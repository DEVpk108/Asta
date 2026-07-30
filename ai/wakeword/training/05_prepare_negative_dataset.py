"""
05_prepare_negative_dataset.py

Prepare and validate negative/background audio for ASTA
wakeword training.

Expected input:
    ai/wakeword/generated/downloaded_negative/

Output:
    ai/wakeword/generated/negative/
        wav/
        negative_manifest.csv
        negative_validation_report.txt

Author: ASTA
"""

from pathlib import Path
import csv
import hashlib
import wave

from tqdm import tqdm

from ai.wakeword.config import GENERATED_DIR


# =========================================================
# Configuration
# =========================================================

SOURCE_DIR = (
    GENERATED_DIR
    / "downloaded_negative"
)

OUTPUT_DIR = (
    GENERATED_DIR
    / "negative"
)

OUTPUT_WAV_DIR = (
    OUTPUT_DIR
    / "wav"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "negative_manifest.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "negative_validation_report.txt"
)

EXPECTED_SAMPLE_RATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2  # 16-bit PCM

MIN_DURATION_SECONDS = 0.20
MAX_DURATION_SECONDS = 10.0

SILENCE_THRESHOLD = 300

CLIPPING_THRESHOLD = 32760

CLIPPING_RATIO_WARNING = 0.001

MIN_CLIPPED_SAMPLES_WARNING = 10


# =========================================================
# WAV analysis
# =========================================================

def analyze_wav(path: Path):

    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        raw_audio = wav.readframes(frames)

    duration = (
        frames / sample_rate
        if sample_rate > 0
        else 0
    )

    import array

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
        "frames": frames,
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "clipped_samples": clipped_samples,
        "clipping_ratio": clipping_ratio,
    }


# =========================================================
# Hash
# =========================================================

def file_hash(path: Path) -> str:

    sha = hashlib.sha256()

    with path.open("rb") as file:

        while chunk := file.read(
            1024 * 1024
        ):

            sha.update(chunk)

    return sha.hexdigest()


# =========================================================
# Manifest
# =========================================================

def write_manifest(rows):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "source_file",
            "duration_seconds",
            "sample_rate",
            "channels",
            "sample_width",
            "peak",
            "rms",
            "sha256",
        ])

        writer.writerows(rows)


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 50)
    print("ASTA NEGATIVE DATASET PREPARATION")
    print("=" * 50)

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            "Negative source directory not found:\n"
            f"{SOURCE_DIR}\n\n"
            "Place your downloaded/background WAV files "
            "there before running this script."
        )

    source_files = sorted(
        SOURCE_DIR.rglob("*.wav")
    )

    if not source_files:

        raise RuntimeError(
            "No negative WAV files found in:\n"
            f"{SOURCE_DIR}"
        )

    print(
        f"Source WAV files : "
        f"{len(source_files)}"
    )

    print(
        f"Output           : "
        f"{OUTPUT_WAV_DIR}"
    )

    print(
        f"Manifest         : "
        f"{MANIFEST_PATH}"
    )

    print("=" * 50)
    print()

    OUTPUT_WAV_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid = 0
    invalid = 0
    duplicates = 0
    warnings = 0

    rows = []

    hashes = {}

    errors = []
    warning_messages = []

    progress = tqdm(
        source_files,
        desc="Preparing",
        unit="clip",
    )

    for index, source_path in enumerate(
        progress,
        start=1,
    ):

        source_relative = (
            source_path
            .relative_to(
                SOURCE_DIR
            )
            .as_posix()
        )

        try:

            info = analyze_wav(
                source_path
            )

        except Exception as exc:

            invalid += 1

            errors.append(
                f"{source_relative}: "
                f"cannot read WAV: {exc}"
            )

            continue

        file_has_error = False

        # -------------------------------------------------
        # Sample rate
        # -------------------------------------------------

        if (
            info["sample_rate"]
            != EXPECTED_SAMPLE_RATE
        ):

            invalid += 1
            file_has_error = True

            errors.append(
                f"{source_relative}: "
                f"sample rate "
                f"{info['sample_rate']} Hz "
                f"(expected "
                f"{EXPECTED_SAMPLE_RATE} Hz)"
            )

        # -------------------------------------------------
        # Channels
        # -------------------------------------------------

        if (
            info["channels"]
            != EXPECTED_CHANNELS
        ):

            invalid += 1
            file_has_error = True

            errors.append(
                f"{source_relative}: "
                f"{info['channels']} channels "
                f"(expected mono)"
            )

        # -------------------------------------------------
        # Sample width
        # -------------------------------------------------

        if (
            info["sample_width"]
            != EXPECTED_SAMPLE_WIDTH
        ):

            invalid += 1
            file_has_error = True

            errors.append(
                f"{source_relative}: "
                f"{info['sample_width'] * 8}-bit "
                f"(expected 16-bit)"
            )

        # -------------------------------------------------
        # Duration
        # -------------------------------------------------

        if (
            info["duration"]
            < MIN_DURATION_SECONDS
        ):

            invalid += 1
            file_has_error = True

            errors.append(
                f"{source_relative}: "
                f"too short "
                f"{info['duration']:.3f}s"
            )

        elif (
            info["duration"]
            > MAX_DURATION_SECONDS
        ):

            warnings += 1

            warning_messages.append(
                f"{source_relative}: "
                f"long clip "
                f"{info['duration']:.3f}s"
            )

        # -------------------------------------------------
        # Silence
        # -------------------------------------------------

        if (
            info["rms"]
            < SILENCE_THRESHOLD
        ):

            warnings += 1

            warning_messages.append(
                f"{source_relative}: "
                f"very quiet "
                f"(RMS={info['rms']:.1f})"
            )

        # -------------------------------------------------
        # Clipping
        # -------------------------------------------------

        if (
            info["clipped_samples"]
            >= MIN_CLIPPED_SAMPLES_WARNING
            and
            info["clipping_ratio"]
            >= CLIPPING_RATIO_WARNING
        ):

            warnings += 1

            warning_messages.append(
                f"{source_relative}: "
                f"possible clipping "
                f"(peak={info['peak']}, "
                f"clipped="
                f"{info['clipped_samples']}, "
                f"ratio="
                f"{info['clipping_ratio'] * 100:.4f}%)"
            )

        # -------------------------------------------------
        # Stop if invalid
        # -------------------------------------------------

        if file_has_error:

            continue

        # -------------------------------------------------
        # Duplicate detection
        # -------------------------------------------------

        digest = file_hash(
            source_path
        )

        if digest in hashes:

            duplicates += 1

            warning_messages.append(
                f"{source_relative}: "
                f"duplicate of "
                f"{hashes[digest]}"
            )

            continue

        hashes[digest] = (
            source_relative
        )

        # -------------------------------------------------
        # Copy into prepared dataset
        # -------------------------------------------------

        output_filename = (
            f"negative_{valid:06d}.wav"
        )

        output_path = (
            OUTPUT_WAV_DIR
            / output_filename
        )

        output_path.write_bytes(
            source_path.read_bytes()
        )

        output_relative = (
            output_path
            .relative_to(
                OUTPUT_DIR
            )
            .as_posix()
        )

        rows.append([
            output_relative,
            source_relative,
            f"{info['duration']:.6f}",
            info["sample_rate"],
            info["channels"],
            info["sample_width"],
            info["peak"],
            f"{info['rms']:.3f}",
            digest,
        ])

        valid += 1

    # -----------------------------------------------------
    # Manifest
    # -----------------------------------------------------

    write_manifest(
        rows
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    report = []

    report.append(
        "ASTA NEGATIVE DATASET "
        "PREPARATION REPORT"
    )

    report.append("=" * 50)

    report.append(
        f"Source WAV files : "
        f"{len(source_files)}"
    )

    report.append(
        f"Valid files      : "
        f"{valid}"
    )

    report.append(
        f"Invalid files    : "
        f"{invalid}"
    )

    report.append(
        f"Duplicates       : "
        f"{duplicates}"
    )

    report.append(
        f"Warnings         : "
        f"{warnings}"
    )

    report.append("")

    if errors:

        report.append(
            "ERRORS"
        )

        report.append(
            "-" * 50
        )

        report.extend(
            errors
        )

        report.append("")

    if warning_messages:

        report.append(
            "WARNINGS"
        )

        report.append(
            "-" * 50
        )

        report.extend(
            warning_messages
        )

        report.append("")

    if invalid == 0:

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
    print("NEGATIVE DATASET PREPARATION COMPLETE")
    print("=" * 50)

    print(
        f"Source WAV files : "
        f"{len(source_files)}"
    )

    print(
        f"Valid            : "
        f"{valid}"
    )

    print(
        f"Invalid          : "
        f"{invalid}"
    )

    print(
        f"Duplicates       : "
        f"{duplicates}"
    )

    print(
        f"Warnings         : "
        f"{warnings}"
    )

    print(
        f"Manifest         : "
        f"{MANIFEST_PATH}"
    )

    print(
        f"Report           : "
        f"{REPORT_PATH}"
    )

    if invalid == 0:

        print()
        print(
            "RESULT: PASS"
        )

        print(
            "Negative dataset is ready "
            "for the next stage."
        )

    else:

        print()
        print(
            "RESULT: FAIL"
        )

        print(
            "Fix invalid negative files "
            "before continuing."
        )

    print("=" * 50)


if __name__ == "__main__":
    main()