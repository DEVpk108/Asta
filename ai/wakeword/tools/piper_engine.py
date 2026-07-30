"""
piper_engine.py

Low-level wrapper around the Piper CLI.

Responsibilities:
    - Verify Piper installation
    - Verify voice model
    - Generate speech from text
"""

from pathlib import Path
import shutil
import subprocess


class PiperEngine:
    """
    Wrapper around the Piper executable.
    """

    def __init__(
        self,
        executable: Path,
        model: Path,
        config: Path,
    ):

        self.executable = Path(executable)
        self.model = Path(model)
        self.config = Path(config)

        self._validate()

    def _validate(self):

        if not self.executable.exists():
            raise FileNotFoundError(
                f"Piper executable not found:\n{self.executable}"
            )

        if not self.model.exists():
            raise FileNotFoundError(
                f"Piper model not found:\n{self.model}"
            )

        if not self.config.exists():
            raise FileNotFoundError(
                f"Piper config not found:\n{self.config}"
            )

    def speak(
        self,
        text: str,
        output_path: Path,
        speaker_id: int | None = None,
    ):
        

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            str(self.executable),
            "--model",
            str(self.model),
            "--config",
            str(self.config),
            "--output_file",
            str(output_path),
        ]
        
        
        if speaker_id is not None:
            command.extend([
            "--speaker",
            str(speaker_id),
        ])
        

        process = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Piper failed.\n\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "Piper completed but output file was not created."
            )

        return output_path

    def version(self) -> str:
        """
        Return Piper version string.
        """

        process = subprocess.run(
            [str(self.executable), "--version"],
            capture_output=True,
            text=True,
        )

        return process.stdout.strip()