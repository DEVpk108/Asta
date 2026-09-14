import os
import shutil
import subprocess
import sys
from pathlib import Path


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
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
        candidates.extend([
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "Scripts" / "python.exe",
        ])
    else:
        candidates.extend([
            root / ".venv" / "bin" / "python",
            root / "venv" / "bin" / "python",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(sys.executable if not getattr(sys, "frozen", False) else "python")


def find_npm() -> str | None:
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm")
    return shutil.which("npm")


def show_error(message: str) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "A.S.T.A.", 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> int:
    root = project_root()
    main_py = root / "main.py"
    hud_dir = root / "hud"
    package_json = hud_dir / "package.json"
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

    if not package_json.exists():
        show_error(f"A.S.T.A. HUD package was not found:\n\n{package_json}")
        return 1

    npm = find_npm()
    if not npm:
        show_error("Node.js / npm was not found. A.S.T.A. requires npm for the HUD.")
        return 1

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["ASTA_PRELAUNCHED_HUD"] = "1"

    hud_process = None
    python_process = None

    try:
        # Start Electron immediately so its renderer can initialize while the
        # Python/AI stack is importing. main.py will reuse this HUD instead of
        # starting a second Electron process.
        hud_process = subprocess.Popen(
            [npm, "start"],
            cwd=str(hud_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            shell=False,
        )

        python_process = subprocess.Popen(
            [str(python_exe), str(main_py)],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            shell=False,
        )

        return_code = python_process.wait()
        return return_code
    except OSError as exc:
        show_error(f"Could not start A.S.T.A.\n\n{type(exc).__name__}: {exc}")
        return 1
    finally:
        if python_process is not None and python_process.poll() is None:
            python_process.terminate()
        if hud_process is not None and hud_process.poll() is None:
            try:
                hud_process.terminate()
                hud_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                hud_process.kill()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
