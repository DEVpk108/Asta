"""Compatibility CLI forwarding to the production 3000-sample preparer."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("05_prepare_negative_dataset.py")), run_name="__main__")
