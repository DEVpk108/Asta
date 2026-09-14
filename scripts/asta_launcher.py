import os
import subprocess
import sys
from pathlib import Path


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


def project_root() -> Path:
    """Resolve the A.S.T.A. project directory next to the launcher executable/script."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def find_python(root: Path) -> Path:
    """Prefer the project's virtual environment, then fall back to PATH Python."""
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                root / ".venv" / "Scripts" / "python.exe",
                root / "venv" / "Scripts" / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                root / ".venv" / "bin" / "python",
                root / "venv" / "bin" / "python",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(sys.executable if not getattr(sys, "frozen", False) else "python")


def show_error(message: str) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            "A.S.T.A.",
            0x10,
        )
    else:
        print(message, file=sys.stderr)


def main() -> int:
    root = project_root()
    main_py = root / "main.py"
    python_exe = find_python(root)

    if not main_py.exists():
        show_error(f"A.S.T.A. runtime not found:\n\n{main_py}")
        return 1

    if not python_exe.exists():
        show_error(
            "A.S.T.A. Python environment was not found.\n\n"
            "Expected .venv\\Scripts\\python.exe next to ASTA.exe."
        )
        return 1

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        subprocess.Popen(
            [str(python_exe), str(main_py)],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                CREATE_NO_WINDOW
                | DETACHED_PROCESS
                | CREATE_NEW_PROCESS_GROUP
            ),
            close_fds=True,
        )
    except OSError as exc:
        show_error(f"Could not start A.S.T.A.\n\n{type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
