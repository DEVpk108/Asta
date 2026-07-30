from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import argparse

SUPPORTED_EXTENSIONS = [".m4a"]


def convert_file(src: Path, dst: Path):
    audio = AudioSegment.from_file(src)

    audio = (
        audio
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )

    dst.parent.mkdir(parents=True, exist_ok=True)

    audio.export(
        dst,
        format="wav",
        codec="pcm_s16le"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASTA negative dataset from M4A to WAV"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delete-original", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    files = []

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))

    print(f"\nFound {len(files)} M4A files.\n")

    converted = 0
    skipped = 0
    failed = 0

    for src in tqdm(files):

        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(".wav")

        if dst.exists():
            skipped += 1
            continue

        try:
            convert_file(src, dst)
            converted += 1

            if args.delete_original:
                src.unlink()

        except Exception as e:
            failed += 1
            print(f"\nFailed: {src}")
            print(e)

    print("\n==============================")
    print("Conversion Report")
    print("==============================")
    print(f"Converted : {converted}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")


if __name__ == "__main__":
    main()