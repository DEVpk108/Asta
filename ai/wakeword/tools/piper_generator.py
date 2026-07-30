"""
piper_generator.py

Generate clean synthetic wakeword recordings using Piper
and maintain a dataset manifest.

Author: ASTA
"""

from pathlib import Path
import csv
import random
import re

from tqdm import tqdm

from ai.wakeword.config import (
    GENERATED_DIR,
    PIPER_EXECUTABLE,
    PIPER_MODEL,
    PIPER_CONFIG,
    WAKEWORD_VARIANTS,
    SPEAKER_WEIGHTS,
)

from ai.wakeword.tools.piper_engine import PiperEngine


class PiperGenerator:

    def __init__(self):

        self.engine = PiperEngine(
            executable=PIPER_EXECUTABLE,
            model=PIPER_MODEL,
            config=PIPER_CONFIG,
        )

        self.output_root = (
            GENERATED_DIR
            / "synthetic_positive"
            / "clean"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Manifest lives beside the clean dataset.
        self.manifest_path = (
            self.output_root
            / "manifest.csv"
        )

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        """
        Convert a phrase into a safe folder name.

        Example:
            "Hey ASTA" -> "hey_asta"
        """

        text = text.lower().strip()

        text = re.sub(
            r"[^a-z0-9]+",
            "_",
            text,
        )

        return text.strip("_")

    # ---------------------------------------------------------

    @staticmethod
    def _choose_variant(variants):
        """
        Select a wakeword variant using configured weights.

        Expected format:

        [
            {
                "text": "Hey ASTA",
                "weight": 50,
            },
            {
                "text": "Hey ASTA!",
                "weight": 10,
            },
        ]
        """

        if not variants:
            raise ValueError(
                "No wakeword variants were provided."
            )

        texts = [
            item["text"]
            for item in variants
        ]

        weights = [
            item["weight"]
            for item in variants
        ]

        return random.choices(
            population=texts,
            weights=weights,
            k=1,
        )[0]

    # ---------------------------------------------------------

    @staticmethod
    def _choose_speaker():
        """
        Select an L2 Arctic speaker using configured weights.
        """

        if not SPEAKER_WEIGHTS:
            raise ValueError(
                "SPEAKER_WEIGHTS is empty."
            )

        speakers = list(
            SPEAKER_WEIGHTS.keys()
        )

        weights = list(
            SPEAKER_WEIGHTS.values()
        )

        return random.choices(
            population=speakers,
            weights=weights,
            k=1,
        )[0]

    # ---------------------------------------------------------

    def _ensure_manifest(self):
        """
        Create the manifest if it doesn't already exist.
        """

        if self.manifest_path.exists():
            return

        with self.manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "filename",
                "canonical_phrase",
                "text",
                "speaker_id",
                "speaker_weight",
            ])

    # ---------------------------------------------------------

    def _manifest_contains(self, filename):
        """
        Check whether a file is already recorded in the manifest.

        This prevents duplicate manifest rows when generation
        is resumed.
        """

        if not self.manifest_path.exists():
            return False

        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["filename"] == filename:
                    return True

        return False

    # ---------------------------------------------------------

    def _write_manifest_entry(
        self,
        filename,
        canonical_phrase,
        text,
        speaker_id,
    ):
        """
        Append one generated clip to the manifest.
        """

        speaker_weight = SPEAKER_WEIGHTS[
            speaker_id
        ]

        with self.manifest_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                filename,
                canonical_phrase,
                text,
                speaker_id,
                speaker_weight,
            ])

    # ---------------------------------------------------------

    def generate(
        self,
        phrases=None,
        samples_per_phrase=1000,
        overwrite=False,
    ):
        """
        Generate clean synthetic wakeword recordings.

        Args:
            phrases:
                Optional list of canonical wakeword phrases.

            samples_per_phrase:
                Number of samples per canonical phrase.

            overwrite:
                If False, existing WAV files are skipped.

                If True, existing WAV files are regenerated.
        """

        if phrases is None:

            phrases = list(
                WAKEWORD_VARIANTS.keys()
            )

        if not phrases:

            raise ValueError(
                "No wakeword phrases configured."
            )

        self._ensure_manifest()

        total = (
            len(phrases)
            * samples_per_phrase
        )

        print(
            "\n"
            "========================================\n"
            "ASTA SYNTHETIC DATASET GENERATION\n"
            "========================================"
        )

        print(
            f"Phrases            : {len(phrases)}"
        )

        print(
            f"Samples / phrase   : "
            f"{samples_per_phrase}"
        )

        print(
            f"Total target       : {total}"
        )

        print(
            f"Speakers           : "
            f"{len(SPEAKER_WEIGHTS)}"
            " configured"
        )

        print(
            f"Manifest           : "
            f"{self.manifest_path}"
        )

        print(
            f"Output             : "
            f"{self.output_root}"
        )

        print(
            "========================================\n"
        )

        generated = 0
        skipped = 0

        progress = tqdm(
            total=total,
            desc="Generating",
            unit="clip",
        )

        for phrase in phrases:

            if phrase not in WAKEWORD_VARIANTS:

                raise KeyError(
                    f"No variants configured for "
                    f"phrase: {phrase!r}"
                )

            variants = (
                WAKEWORD_VARIANTS[phrase]
            )

            folder = (
                self.output_root
                / self._slugify(phrase)
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            for index in range(
                samples_per_phrase
            ):

                speaker = (
                    self._choose_speaker()
                )

                text = (
                    self._choose_variant(
                        variants
                    )
                )

                filename = (
                    folder
                    / f"{speaker:02d}_"
                    f"{index:05d}.wav"
                )

                relative_filename = (
                    filename.relative_to(
                        self.output_root
                    )
                    .as_posix()
                )

                # -------------------------------------------------
                # Existing file
                # -------------------------------------------------

                if (
                    filename.exists()
                    and not overwrite
                ):

                    skipped += 1

                    # Make sure an existing WAV has
                    # a manifest entry.
                    if not self._manifest_contains(
                        relative_filename
                    ):

                        self._write_manifest_entry(
                            filename=relative_filename,
                            canonical_phrase=phrase,
                            text=text,
                            speaker_id=speaker,
                        )

                    progress.update(1)

                    continue

                # -------------------------------------------------
                # Generate WAV
                # -------------------------------------------------

                self.engine.speak(
                    text=text,
                    speaker_id=speaker,
                    output_path=filename,
                )

                # -------------------------------------------------
                # Record metadata
                # -------------------------------------------------

                self._write_manifest_entry(
                    filename=relative_filename,
                    canonical_phrase=phrase,
                    text=text,
                    speaker_id=speaker,
                )

                generated += 1

                progress.update(1)

        progress.close()

        print(
            "\n"
            "========================================\n"
            "GENERATION COMPLETE\n"
            "========================================"
        )

        print(
            f"Generated : {generated}"
        )

        print(
            f"Skipped   : {skipped}"
        )

        print(
            f"Total     : "
            f"{generated + skipped}"
        )

        print(
            f"Manifest  : "
            f"{self.manifest_path}"
        )

        print(
            "========================================\n"
        )
    def reset_manifest(self):
        
        """
        Delete the existing manifest.

        Use this before starting a fresh production dataset.
        """

        if self.manifest_path.exists():
            self.manifest_path.unlink()

            print(
                f"Manifest reset: {self.manifest_path}"
            )
        else:
            print("No manifest to reset.")