import csv
import wave
from pathlib import Path

import numpy as np

from . import config


class MetadataManager:

    def __init__(self):

        self.sample_rate = config.SAMPLE_RATE
        self.metadata_file = config.METADATA_FILE

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.metadata_file.exists():

            with open(self.metadata_file, "w", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "filename",
                    "text",
                    "duration",
                    "sample_rate",
                    "samples",
                ])

        self.counter = self._load_counter()

    # --------------------------------------------------

    def _load_counter(self):

        if not self.metadata_file.exists():
            return 1

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        rows = [r for r in rows if len(r) > 0]

        if len(rows) <= 1:
            return 1

        try:
            number = int(Path(rows[-1][0]).stem)
            return number + 1
        except Exception:
            return 1

    # --------------------------------------------------

    def _next_filename(self):

        filename = f"{self.counter:06d}.wav"
        self.counter += 1

        return filename

    # --------------------------------------------------

    def save(self, audio: np.ndarray, text: str):

        filename = self._next_filename()

        folder = config.POSITIVE_FOLDERS[text]

        filepath = folder / filename

        self._save_audio(filepath, audio)

        duration = len(audio) / self.sample_rate

        relative_path = filepath.relative_to(config.DATASET_DIR)

        with open(
            self.metadata_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                str(relative_path),
                text,
                round(duration, 2),
                self.sample_rate,
                len(audio),
            ])

        return filepath

    # --------------------------------------------------

    def _save_audio(self, filepath, audio):

        audio = np.clip(audio, -1.0, 1.0)

        pcm = (audio * 32767).astype(np.int16)

        with wave.open(str(filepath), "wb") as wf:

            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())