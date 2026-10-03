"""Download the low-latency speech models used by A.S.T.A.

    python scripts/download_speech_models.py            # Nemotron 560 ms + Smart Turn
    python scripts/download_speech_models.py --chunk 160 # lower partial latency

Models are stored under models/speech/ (git-ignored):
  * NVIDIA Nemotron 3.5 ASR Streaming 0.6B (sherpa-onnx int8 export),
    English + Hindi + 38 more language-locales, CC-BY-4.0 / NVIDIA license.
  * pipecat-ai Smart Turn v3.2 (CPU ONNX), BSD-2-Clause.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "speech"
NEMOTRON = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{chunk}ms-int8-2026-06-11.tar.bz2"
)
SMART_TURN = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
CHUNKS = (80, 160, 320, 560, 1120)


def _download(url: str, destination: Path) -> None:
    print(f"Downloading {url}")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(100, blocks * block_size * 100 // total)
            print(f"\r  {done:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=progress)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk", type=int, default=560, choices=CHUNKS,
                        help="Nemotron streaming chunk in ms (smaller = faster partials, slightly less accurate)")
    parser.add_argument("--skip-smart-turn", action="store_true")
    args = parser.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    name = f"sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-{args.chunk}ms-int8-2026-06-11"
    if (TARGET / name / "tokens.txt").is_file():
        print(f"Nemotron already installed: {TARGET / name}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "nemotron.tar.bz2"
            _download(NEMOTRON.format(chunk=args.chunk), archive)
            print("Extracting...")
            with tarfile.open(archive, "r:bz2") as tar:
                try:
                    tar.extractall(TARGET, filter="data")
                except TypeError:  # Python without extraction filters
                    tar.extractall(TARGET)
        print(f"Nemotron installed: {TARGET / name}")

    if not args.skip_smart_turn:
        smart_turn = TARGET / "smart-turn-v3.2-cpu.onnx"
        if smart_turn.is_file():
            print(f"Smart Turn already installed: {smart_turn}")
        else:
            _download(SMART_TURN, smart_turn)
            print(f"Smart Turn installed: {smart_turn}")

    if args.chunk != 560:
        print(f"\nSet ASTA_NEMOTRON_MODEL_DIR={TARGET / name} in .env to use this chunk size.")
    print("\nDone. Restart A.S.T.A.; it picks Nemotron automatically (ASTA_STT_BACKEND=auto).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
