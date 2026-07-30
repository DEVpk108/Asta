from pathlib import Path
import argparse
import wave
import numpy as np
from tqdm import tqdm


SUPPORTED = [".wav"]


class Stats:

    def __init__(self):

        self.total = 0
        self.corrupted = 0
        self.empty = 0
        self.silent = 0

        self.sample_rates = {}
        self.channels = {}
        self.bit_depths = {}

        self.total_duration = 0.0


def rms(audio):

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        return 0

    return np.sqrt(np.mean(audio ** 2))


def validate_file(path: Path, stats: Stats):

    stats.total += 1

    try:

        with wave.open(str(path), "rb") as wf:

            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.getnframes()

            duration = frames / sample_rate

            stats.total_duration += duration

            stats.sample_rates[sample_rate] = (
                stats.sample_rates.get(sample_rate, 0) + 1
            )

            stats.channels[channels] = (
                stats.channels.get(channels, 0) + 1
            )

            bit_depth = sample_width * 8

            stats.bit_depths[bit_depth] = (
                stats.bit_depths.get(bit_depth, 0) + 1
            )

            if frames == 0:
                stats.empty += 1
                return

            raw = wf.readframes(frames)

            audio = np.frombuffer(raw, dtype=np.int16)

            if rms(audio) < 5:
                stats.silent += 1

    except Exception:

        stats.corrupted += 1


def scan_folder(folder: Path):

    stats = Stats()

    files = []

    for ext in SUPPORTED:
        files.extend(folder.rglob(f"*{ext}"))

    for file in tqdm(files):

        validate_file(file, stats)

    return stats


def print_stats(name, stats):

    print("\n===================================")
    print(name)
    print("===================================\n")

    print(f"Files : {stats.total}")

    print(f"Corrupted : {stats.corrupted}")
    print(f"Empty     : {stats.empty}")
    print(f"Silent    : {stats.silent}")

    print()

    print("Sample Rates")

    for k, v in sorted(stats.sample_rates.items()):
        print(f"{k} Hz : {v}")

    print()

    print("Channels")

    for k, v in sorted(stats.channels.items()):
        print(f"{k} : {v}")

    print()

    print("Bit Depth")

    for k, v in sorted(stats.bit_depths.items()):
        print(f"{k} bit : {v}")

    print()

    if stats.total:

        avg = stats.total_duration / stats.total

        print(f"Average Duration : {avg:.2f} sec")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    args = parser.parse_args()

    root = Path(args.dataset)

    positive = root / "positive"

    negative = root / "negative_wav"

    pos = scan_folder(positive)

    neg = scan_folder(negative)

    print_stats("POSITIVE DATASET", pos)

    print_stats("NEGATIVE DATASET", neg)

    print("\n===================================")
    print("ASTA DATASET STATUS")
    print("===================================\n")

    if (
        pos.corrupted
        or neg.corrupted
        or pos.empty
        or neg.empty
    ):

        print("DATASET NOT READY")

    else:

        print("DATASET READY FOR TRAINING")


if __name__ == "__main__":
    main()