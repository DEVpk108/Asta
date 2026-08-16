from pathlib import Path
import sys

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SYNTHETIC_ROOT = (
    PROJECT_ROOT
    / "generated"
    / "synthetic_positive"
    / "augmented"
)

REAL_ROOT = (
    PROJECT_ROOT
    / "generated"
    / "real"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "generated"
    / "synthetic_positive"
    / "features"
)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

WAKE_PHRASES = [
    "hello_asta",
    "hey_asta",
    "wake_up_asta",
]

TARGET_SR = 16000

# 2 seconds at 16 kHz.
TARGET_SAMPLES = 32000

BATCH_SIZE = 16
NCPU = 1


# ---------------------------------------------------------------------
# Import openWakeWord
# ---------------------------------------------------------------------

OPENWAKEWORD_ROOT = PROJECT_ROOT / "openWakeWord"

sys.path.insert(0, str(OPENWAKEWORD_ROOT))

from openwakeword.utils import AudioFeatures


# ---------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------

def load_audio(path):
    """
    Load WAV, convert to mono, resample to 16 kHz,
    normalize, and pad/trim to exactly 2 seconds.
    """

    sr, audio = wavfile.read(path)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample to 16 kHz
    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr,
        )

    # Normalize each recording safely.

    audio = np.clip(
        audio,
        -32768,
        32767,
    )

    audio = audio.astype(np.int16)

    # -------------------------------------------------------------
    # Pad short clips.
    #
    # We don't always put the audio exactly in the center.
    # A small random shift gives the embedding extractor
    # slightly different temporal alignment.
    # -------------------------------------------------------------

    if len(audio) < TARGET_SAMPLES:

        remaining = TARGET_SAMPLES - len(audio)

        max_shift = min(
            remaining,
            4000,
        )

        center = remaining // 2

        shift = np.random.randint(
            -max_shift,
            max_shift + 1,
        )

        left = center + shift

        left = max(
            0,
            min(left, remaining),
        )

        right = remaining - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    # -------------------------------------------------------------
    # Trim long clips.
    # -------------------------------------------------------------

    elif len(audio) > TARGET_SAMPLES:

        start = np.random.randint(
            0,
            len(audio) - TARGET_SAMPLES + 1,
        )

        audio = audio[
            start:start + TARGET_SAMPLES
        ]

    return audio


# ---------------------------------------------------------------------
# Get WAV files from both datasets
# ---------------------------------------------------------------------

def get_wav_files(class_name):

    synthetic_dir = SYNTHETIC_ROOT / class_name
    real_dir = REAL_ROOT / class_name

    synthetic_files = sorted(
        synthetic_dir.glob("*.wav")
    ) if synthetic_dir.exists() else []

    real_files = sorted(
        real_dir.glob("*.wav")
    ) if real_dir.exists() else []

    if not synthetic_files:
        raise FileNotFoundError(
            f"No synthetic WAV files found:\n"
            f"{synthetic_dir}"
        )

    if not real_files:
        raise FileNotFoundError(
            f"No real WAV files found:\n"
            f"{real_dir}"
        )

    # -------------------------------------------------------------
    # Keep source information so the final report can tell us
    # exactly what went into the dataset.
    # -------------------------------------------------------------

    print()
    print(f"Synthetic WAV files : {len(synthetic_files)}")
    print(f"Real WAV files      : {len(real_files)}")

    wav_files = (
        synthetic_files +
        real_files
    )

    # Shuffle the combined dataset.
    #
    # This prevents all synthetic samples from being processed
    # first and all real samples afterwards.
    #
    # We shuffle paths, not audio data.
    rng = np.random.default_rng(42)
    rng.shuffle(wav_files)

    return wav_files, len(synthetic_files), len(real_files)


# ---------------------------------------------------------------------
# Process one wake phrase
# ---------------------------------------------------------------------

def extract_class_features(
    features,
    class_name,
):

    wav_files, synthetic_count, real_count = (
        get_wav_files(class_name)
    )

    print()
    print("=" * 70)
    print(f"WAKE PHRASE: {class_name}")
    print("=" * 70)
    print(f"Synthetic : {synthetic_count}")
    print(f"Real      : {real_count}")
    print(f"TOTAL     : {len(wav_files)}")
    print("=" * 70)

    batches = []

    for start in range(
        0,
        len(wav_files),
        BATCH_SIZE,
    ):

        batch_paths = wav_files[
            start:start + BATCH_SIZE
        ]

        audio_batch = []

        for path in batch_paths:

            try:

                audio = load_audio(path)

                audio_batch.append(audio)

            except Exception as exc:

                print()
                print(
                    f"WARNING: Could not process:"
                )
                print(f"  {path}")
                print(f"Reason: {exc}")

        if not audio_batch:
            continue

        audio_batch = np.asarray(
            audio_batch,
            dtype=np.int16,
        )

        # -------------------------------------------------------------
        # Compute openWakeWord embeddings.
        # -------------------------------------------------------------

        batch_features = features.embed_clips(
            audio_batch,
            batch_size=len(audio_batch),
            ncpu=NCPU,
        )

        batches.append(
            batch_features
        )

        processed = min(
            start + len(batch_paths),
            len(wav_files),
        )

        print(
            f"\rProcessed "
            f"{processed}/{len(wav_files)}",
            end="",
            flush=True,
        )

    print()

    if not batches:
        raise RuntimeError(
            f"No features were generated for {class_name}"
        )

    result = np.concatenate(
        batches,
        axis=0,
    ).astype(np.float32)

    print(
        f"Feature shape: {result.shape}"
    )

    return result, synthetic_count, real_count


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("ASTA POSITIVE FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Synthetic source:\n"
        f"  {SYNTHETIC_ROOT.resolve()}"
    )

    print(
        f"Real source:\n"
        f"  {REAL_ROOT.resolve()}"
    )

    print(
        f"Output directory:\n"
        f"  {OUTPUT_DIR.resolve()}"
    )

    print()
    print("Wake phrases:")

    for phrase in WAKE_PHRASES:
        print(f"  - {phrase}")

    print()
    print(f"Target sample rate : {TARGET_SR} Hz")
    print(
        f"Target duration    : "
        f"{TARGET_SAMPLES / TARGET_SR:.1f} seconds"
    )
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"CPU workers        : {NCPU}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Initialize AudioFeatures
    # -----------------------------------------------------------------

    print()
    print(
        "Initializing openWakeWord "
        "AudioFeatures..."
    )

    features = AudioFeatures(
        device="cpu",
        ncpu=NCPU,
    )

    # -----------------------------------------------------------------
    # Extract each phrase separately
    # -----------------------------------------------------------------

    class_features = {}
    dataset_counts = {}

    for phrase in WAKE_PHRASES:

        (
            class_features[phrase],
            synthetic_count,
            real_count,
        ) = extract_class_features(
            features,
            phrase,
        )

        dataset_counts[phrase] = {
            "synthetic": synthetic_count,
            "real": real_count,
        }

    # -----------------------------------------------------------------
    # Save individual datasets
    # -----------------------------------------------------------------

    for phrase, data in class_features.items():

        output_file = (
            OUTPUT_DIR
            / f"{phrase}_features.npy"
        )

        np.save(
            output_file,
            data,
        )

        print()
        print(
            f"Saved {phrase}: "
            f"{output_file}"
        )

        print(
            f"Shape: {data.shape}"
        )

    # -----------------------------------------------------------------
    # Combine all three phrases
    # -----------------------------------------------------------------

    positive_features = np.concatenate(
        [
            class_features["hello_asta"],
            class_features["hey_asta"],
            class_features["wake_up_asta"],
        ],
        axis=0,
    ).astype(np.float32)

    positive_output = (
        OUTPUT_DIR
        / "positive_features.npy"
    )

    np.save(
        positive_output,
        positive_features,
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("COMBINED POSITIVE DATASET")
    print("=" * 70)

    total_synthetic = 0
    total_real = 0

    for phrase in WAKE_PHRASES:

        synthetic = dataset_counts[phrase]["synthetic"]
        real = dataset_counts[phrase]["real"]

        total_synthetic += synthetic
        total_real += real

        print(
            f"{phrase:<15} "
            f"synthetic={synthetic:<5} "
            f"real={real:<5} "
            f"total={synthetic + real}"
        )

    print("-" * 70)

    print(
        f"TOTAL SYNTHETIC : "
        f"{total_synthetic}"
    )

    print(
        f"TOTAL REAL      : "
        f"{total_real}"
    )

    print(
        f"TOTAL POSITIVE  : "
        f"{len(positive_features)}"
    )

    print(
        f"Shape           : "
        f"{positive_features.shape}"
    )

    print(
        f"Saved to        : "
        f"{positive_output}"
    )

    print("=" * 70)
    print(
        "FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()