from pathlib import Path
import json
import wave
import numpy as np
from datetime import datetime


SUPPORTED = [".wav"]


def scan(folder: Path):

    stats = {
        "files": 0,
        "duration": 0,
        "sample_rates": {},
        "channels": {},
        "bit_depths": {},
        "average_duration": 0
    }

    wav_files = []

    for ext in SUPPORTED:
        wav_files.extend(folder.rglob(f"*{ext}"))

    for file in wav_files:

        with wave.open(str(file), "rb") as wf:

            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth() * 8

            frames = wf.getnframes()

            duration = frames / sr

            stats["files"] += 1
            stats["duration"] += duration

            stats["sample_rates"][str(sr)] = (
                stats["sample_rates"].get(str(sr), 0) + 1
            )

            stats["channels"][str(ch)] = (
                stats["channels"].get(str(ch), 0) + 1
            )

            stats["bit_depths"][str(sw)] = (
                stats["bit_depths"].get(str(sw), 0) + 1
            )

    if stats["files"]:

        stats["average_duration"] = round(
            stats["duration"] / stats["files"],
            2
        )

    stats["duration"] = round(stats["duration"], 2)

    return stats


def main():

    # Directory where this script lives
    script_dir = Path(__file__).resolve().parent

    # Dataset root (wakeword_dataset_v1)
    root = script_dir.parent

    positive = root / "positive"
    negative = root / "negative_wav"
    
    print(f"Dataset Root : {root}")
    print(f"Positive     : {positive}")
    print(f"Negative     : {negative}")
    print(f"Positive Exists : {positive.exists()}")
    print(f"Negative Exists : {negative.exists()}")

    info = {

        "dataset_name": "ASTA Wakeword Dataset",

        "dataset_version": "v1.0",

        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "positive": scan(positive),

        "negative": scan(negative),

        "training_status": "READY",

        "recommended_framework": "openWakeWord",

        "audio_format": {

            "sample_rate": 16000,

            "channels": 1,

            "bit_depth": 16,

            "encoding": "PCM"

        }

    }
    
    output = root / "dataset_info.json"

    with open(output, "w", encoding="utf-8") as f:

        json.dump(
            info,
            f,
            indent=4
        )

    print()

    print("\nDataset information generated.")

    print(output)


if __name__ == "__main__":

    main()