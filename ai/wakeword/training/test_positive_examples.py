from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import onnxruntime as ort

from openwakeword.utils import AudioFeatures


TARGET_SR = 16000
TARGET_SAMPLES = 32000

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "models"
)

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "synthetic_positive"
    / "augmented"
)

REAL_DIR = (
    PROJECT_ROOT
    / "ai"
    / "wakeword"
    / "generated"
    / "real"
)


def load_audio(path):
    sr, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        audio = resample_poly(
            audio,
            TARGET_SR,
            sr,
        )

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    audio *= 30000
    audio = audio.astype(np.int16)

    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(
            audio,
            (0, TARGET_SAMPLES - len(audio)),
            mode="constant",
        )
    elif len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]

    return audio


def score_file(model, features, path):
    audio = load_audio(path)

    embedding = features.embed_clips(
        np.asarray([audio], dtype=np.int16),
        batch_size=1,
        ncpu=1,
    )

    input_name = model.get_inputs()[0].name

    result = model.run(
        None,
        {
            input_name: embedding,
        },
    )

    return float(np.asarray(result[0]).max())


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phrase",
        required=True,
        choices=[
            "hello_asta",
            "hey_asta",
            "wake_up_asta",
        ],
    )

    args = parser.parse_args()
    phrase = args.phrase

    model_path = MODEL_DIR / f"{phrase}.onnx"
    synthetic_dir = SYNTHETIC_DIR / phrase
    real_dir = REAL_DIR / phrase

    print("=" * 70)
    print(f"POSITIVE DATASET DIAGNOSTIC: {phrase}")
    print("=" * 70)

    print("Model:")
    print(model_path)

    print()
    print("Synthetic:")
    print(synthetic_dir)

    print()
    print("Real:")
    print(real_dir)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    synthetic_files = sorted(
        synthetic_dir.glob("*.wav")
    )[:10]

    real_files = sorted(
        real_dir.glob("*.wav")
    )[:10]

    print()
    print(f"Synthetic samples: {len(synthetic_files)}")
    print(f"Real samples     : {len(real_files)}")

    if not synthetic_files:
        raise RuntimeError(
            f"No synthetic WAV files found in {synthetic_dir}"
        )

    if not real_files:
        raise RuntimeError(
            f"No real WAV files found in {real_dir}"
        )

    print()
    print("Loading model...")

    model = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    print("Initializing AudioFeatures...")

    features = AudioFeatures(
        device="cpu",
        ncpu=1,
    )

    print()
    print("=" * 70)
    print("SYNTHETIC POSITIVES")
    print("=" * 70)

    synthetic_scores = []

    for i, path in enumerate(synthetic_files, 1):

        score = score_file(
            model,
            features,
            path,
        )

        synthetic_scores.append(score)

        print(
            f"{i:2d}/10 "
            f"{path.name:<35} "
            f"score={score:.6f}"
        )

    print()
    print("=" * 70)
    print("REAL POSITIVES")
    print("=" * 70)

    real_scores = []

    for i, path in enumerate(real_files, 1):

        score = score_file(
            model,
            features,
            path,
        )

        real_scores.append(score)

        print(
            f"{i:2d}/10 "
            f"{path.name:<35} "
            f"score={score:.6f}"
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Synthetic mean : {np.mean(synthetic_scores):.6f}"
    )

    print(
        f"Synthetic max  : {np.max(synthetic_scores):.6f}"
    )

    print(
        f"Real mean      : {np.mean(real_scores):.6f}"
    )

    print(
        f"Real max       : {np.max(real_scores):.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()