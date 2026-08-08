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

POSITIVE_ROOT = (
    PROJECT_ROOT
    / "generated"
    / "synthetic_positive"
    / "augmented"
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

# openWakeWord training uses a minimum of 32000 samples
# = 2 seconds at 16 kHz.
TARGET_SAMPLES = 32000

# Keep this conservative on Windows.
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
    """Load WAV, convert to mono, resample to 16 kHz,
    and pad/trim to exactly 2 seconds."""

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

    # Prevent overflow when converting back to int16
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    audio *= 30000

    audio = audio.astype(np.int16)

    # Pad short clips
    if len(audio) < TARGET_SAMPLES:
        remaining = TARGET_SAMPLES - len(audio)

        max_shift = min(remaining, 4000)   # about 250 ms

        left = remaining // 2 + np.random.randint(-max_shift, max_shift + 1)

        left = max(0, min(left, remaining))
        right = remaining - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant"
        )

    # Trim long clips
    elif len(audio) > TARGET_SAMPLES:
        start = np.random.randint(0, len(audio) - TARGET_SAMPLES + 1)
        audio = audio[start:start + TARGET_SAMPLES]

    return audio


# ---------------------------------------------------------------------
# Process one wake phrase
# ---------------------------------------------------------------------

def extract_class_features(
    features,
    class_name,
):

    input_dir = POSITIVE_ROOT / class_name

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Missing wake phrase directory:\n{input_dir}"
        )

    wav_files = sorted(
        input_dir.glob("*.wav")
    )

    if not wav_files:
        raise RuntimeError(
            f"No WAV files found in:\n{input_dir}"
        )

    print()
    print("=" * 70)
    print(f"WAKE PHRASE: {class_name}")
    print(f"WAV COUNT : {len(wav_files)}")
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
                    f"WARNING: Could not process {path}"
                )
                print(f"Reason: {exc}")

        if not audio_batch:
            continue

        audio_batch = np.asarray(
            audio_batch,
            dtype=np.int16,
        )

        # -------------------------------------------------------------
        # Compute openWakeWord embeddings for the entire batch.
        # -------------------------------------------------------------

        batch_features = features.embed_clips(
            audio_batch,
            batch_size=len(audio_batch),
            ncpu=NCPU,
        )

        batches.append(batch_features)

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

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("ASTA POSITIVE FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"Source directory:\n"
        f"  {POSITIVE_ROOT.resolve()}"
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
    print(f"Target duration    : {TARGET_SAMPLES / TARGET_SR:.1f} seconds")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"CPU workers        : {NCPU}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Initializing openWakeWord AudioFeatures...")

    features = AudioFeatures(
        device="cpu",
        ncpu=NCPU,
    )

    # -----------------------------------------------------------------
    # Extract each phrase separately
    # -----------------------------------------------------------------

    class_features = {}

    for phrase in WAKE_PHRASES:

        class_features[phrase] = extract_class_features(
            features,
            phrase,
        )

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
    # Combine all three phrases into ONE positive class
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

    print(
        f"hello asta : "
        f"{len(class_features['hello_asta'])}"
    )

    print(
        f"hey asta   : "
        f"{len(class_features['hey_asta'])}"
    )

    print(
        f"wake up asta: "
        f"{len(class_features['wake_up_asta'])}"
    )

    print(
        f"TOTAL      : "
        f"{len(positive_features)}"
    )

    print(
        f"Shape      : "
        f"{positive_features.shape}"
    )

    print(
        f"Saved to   : "
        f"{positive_output}"
    )

    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()