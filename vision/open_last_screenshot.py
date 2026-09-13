import os
from pathlib import Path


def open_last_screenshot(output_dir: str | os.PathLike = "runtime/screenshots") -> dict:
    directory = Path(output_dir)
    screenshots = sorted(directory.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not screenshots:
        raise FileNotFoundError("No screenshots have been captured yet.")

    path = screenshots[0].resolve()
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        raise RuntimeError("Opening screenshots automatically is currently supported on Windows only.")

    return {"path": str(path), "opened": True}
