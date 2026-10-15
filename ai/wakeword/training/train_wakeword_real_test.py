"""Compatibility entry point; use train_wakeword.py for the canonical trainer."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_wakeword.py")), run_name="__main__")
