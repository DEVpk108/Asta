"""
03_augment_dataset.py

Create augmented positive wakeword samples from the validated
clean ASTA dataset.

Author: ASTA
"""

from pathlib import Path
import csv
import random
import wave

import numpy as np
from tqdm import tqdm

from ai.wakeword.config import GENERATED_DIR


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CLEAN_DIR = (
    GENERATED_DIR
    / "synthetic_positive"
    / "clean"
)

AUGMENTED_DIR = (
    GENERATED_DIR
    / "synthetic_positive"
    / "augmented"
)

AUGMENTED_MANIFEST = (
    AUGMENTED_DIR
    / "augmentation_manifest.csv"
)

TARGET_AUGMENTED_SAMPLES = 5000

SAMPLE_RATE = 22050

# Maximum gain change.
MIN_GAIN_DB = -4.0
MAX_GAIN_DB = 3.0

# Background noise strength.
MIN_NOISE_LEVEL = 0.002
MAX_NOISE_LEVEL = 0.015

# Small speed variation.
MIN_SPEED = 0.95
MAX_SPEED = 1.05


# ---------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------

def read_wav(path: Path):
    """
    Read a mono 16-bit PCM WAV.
    """

    with wave.open(str(path), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        audio = wav.readframes(frames)

    if channels != 1:
        raise ValueError(
            f"{path} is not mono."
        )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample rate "
            f"{sample_rate}, expected "
            f"{SAMPLE_RATE}."
        )

    if sample_width != 2:
        raise ValueError(
            f"{path} is not 16-bit PCM."
        )

    samples = np.frombuffer(
        audio,
        dtype=np.int16,
    ).astype(np.float32)

    return samples


# ---------------------------------------------------------

def write_wav(
    path: Path,
    samples: np.ndarray,
):
    """
    Write mono 16-bit PCM WAV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = np.clip(
        samples,
        -32768,
        32767,
    ).astype(np.int16)

    with wave.open(
        str(path),
        "wb",
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            samples.tobytes()
        )


# ---------------------------------------------------------
# Augmentations
# ---------------------------------------------------------

def apply_gain(
    audio: np.ndarray,
    gain_db: float,
):
    """
    Apply volume change in decibels.
    """

    gain = 10 ** (
        gain_db / 20.0
    )

    return audio * gain


# ---------------------------------------------------------

def add_noise(
    audio: np.ndarray,
    noise_level: float,
):
    """
    Add low-level white noise.
    """

    noise = np.random.normal(
        0,
        32767 * noise_level,
        size=len(audio),
    )

    return audio + noise


# ---------------------------------------------------------

def change_speed(
    audio: np.ndarray,
    speed: float,
):
    """
    Small speed variation using linear interpolation.

    speed > 1.0:
        slightly faster

    speed < 1.0:
        slightly slower
    """

    if len(audio) < 2:
        return audio

    new_length = int(
        len(audio) / speed
    )

    if new_length < 2:
        return audio

    old_positions = np.linspace(
        0,
        len(audio) - 1,
        num=len(audio),
    )

    new_positions = np.linspace(
        0,
        len(audio) - 1,
        num=new_length,
    )

    return np.interp(
        new_positions,
        old_positions,
        audio,
    )


# ---------------------------------------------------------

def random_augmentation(audio):
    """
    Apply a random combination of augmentations.

    Returns:
        augmented_audio,
        metadata
    """

    output = audio.copy()

    metadata = {
        "gain_db": 0.0,
        "noise_level": 0.0,
        "speed": 1.0,
        "augmentations": [],
    }

    # -----------------------------------------------------
    # Gain
    # -----------------------------------------------------

    if random.random() < 0.75:

        gain_db = random.uniform(
            MIN_GAIN_DB,
            MAX_GAIN_DB,
        )

        output = apply_gain(
            output,
            gain_db,
        )

        metadata["gain_db"] = round(
            gain_db,
            3,
        )

        metadata[
            "augmentations"
        ].append("gain")

    # -----------------------------------------------------
    # Noise
    # -----------------------------------------------------

    if random.random() < 0.70:

        noise_level = random.uniform(
            MIN_NOISE_LEVEL,
            MAX_NOISE_LEVEL,
        )

        output = add_noise(
            output,
            noise_level,
        )

        metadata["noise_level"] = round(
            noise_level,
            6,
        )

        metadata[
            "augmentations"
        ].append("noise")

    # -----------------------------------------------------
    # Speed
    # -----------------------------------------------------

    if random.random() < 0.50:

        speed = random.uniform(
            MIN_SPEED,
            MAX_SPEED,
        )

        output = change_speed(
            output,
            speed,
        )

        metadata["speed"] = round(
            speed,
            4,
        )

        metadata[
            "augmentations"
        ].append("speed")

    # -----------------------------------------------------
    # Final safety normalization
    # -----------------------------------------------------

    peak = np.max(
        np.abs(output)
    )

    if peak > 32767:

        output = (
            output
            * (32767 / peak)
        )

        metadata[
            "augmentations"
        ].append("peak_normalization")

    return output, metadata


# ---------------------------------------------------------
# Manifest
# ---------------------------------------------------------

def create_manifest():

    AUGMENTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUGMENTED_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "source_file",
            "gain_db",
            "noise_level",
            "speed",
            "augmentations",
        ])


# ---------------------------------------------------------

def append_manifest(
    filename,
    source_file,
    metadata,
):

    with AUGMENTED_MANIFEST.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            filename,
            source_file,
            metadata["gain_db"],
            metadata["noise_level"],
            metadata["speed"],
            "|".join(
                metadata["augmentations"]
            ),
        ])


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print()
    print("=" * 50)
    print("ASTA DATASET AUGMENTATION")
    print("=" * 50)

    if not CLEAN_DIR.exists():

        raise FileNotFoundError(
            f"Clean dataset not found:\n"
            f"{CLEAN_DIR}"
        )

    clean_files = sorted(
        CLEAN_DIR.rglob("*.wav")
    )

    if not clean_files:

        raise RuntimeError(
            "No clean WAV files found."
        )

    print(
        f"Clean samples       : "
        f"{len(clean_files)}"
    )

    print(
        f"Target augmented    : "
        f"{TARGET_AUGMENTED_SAMPLES}"
    )

    print(
        f"Output              : "
        f"{AUGMENTED_DIR}"
    )

    print(
        f"Manifest            : "
        f"{AUGMENTED_MANIFEST}"
    )

    print("=" * 50)
    print()

    # -----------------------------------------------------
    # Fresh augmentation dataset
    # -----------------------------------------------------

    create_manifest()

    generated = 0

    progress = tqdm(
        total=TARGET_AUGMENTED_SAMPLES,
        desc="Augmenting",
        unit="clip",
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    while generated < TARGET_AUGMENTED_SAMPLES:

        source_path = random.choice(
            clean_files
        )

        audio = read_wav(
            source_path
        )

        augmented, metadata = (
            random_augmentation(
                audio
            )
        )

        source_relative = (
            source_path
            .relative_to(
                CLEAN_DIR
            )
        )

        source_stem = (
            source_path.stem
        )

        output_index = generated

        output_folder = (
            AUGMENTED_DIR
            / source_relative.parent
        )

        output_filename = (
            f"{source_stem}"
            f"_aug_{output_index:05d}.wav"
        )

        output_path = (
            output_folder
            / output_filename
        )

        output_relative = (
            output_path
            .relative_to(
                AUGMENTED_DIR
            )
            .as_posix()
        )

        write_wav(
            output_path,
            augmented,
        )

        append_manifest(
            filename=output_relative,
            source_file=(
                source_relative
                .as_posix()
            ),
            metadata=metadata,
        )

        generated += 1

        progress.update(1)

    progress.close()

    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print()
    print("=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)

    print(
        f"Clean samples       : "
        f"{len(clean_files)}"
    )

    print(
        f"Augmented samples   : "
        f"{generated}"
    )

    print(
        f"Total positive      : "
        f"{len(clean_files) + generated}"
    )

    print(
        f"Manifest            : "
        f"{AUGMENTED_MANIFEST}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()