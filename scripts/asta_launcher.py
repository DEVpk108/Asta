import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
MUTEX_ERROR_ALREADY_EXISTS = 183


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
        ctypes.windll.user32.MessageBoxW(0, message, "A.S.T.A.", 0x10)
    else:
        print(message, file=sys.stderr)


def acquire_single_instance_lock():
    """Prevent multiple packaged launches from loading multiple AI stacks."""
    if os.name != "nt":
        return None

    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\ASTA_Runtime_Instance")
    if not handle:
        return None

    if ctypes.windll.kernel32.GetLastError() == MUTEX_ERROR_ALREADY_EXISTS:
        ctypes.windll.kernel32.CloseHandle(handle)
        show_error("A.S.T.A. is already running.\n\nOnly one A.S.T.A. runtime can run at a time.")
        return False

    return handle


def release_single_instance_lock(handle) -> None:
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)


def terminate_process_tree(process) -> None:
    """Terminate a process and every descendant it spawned."""
    if process is None or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=CREATE_NO_WINDOW,
            )
        else:
            process.terminate()
            process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass
    except OSError:
        pass


def main() -> int:
    mutex = acquire_single_instance_lock()
    if mutex is False:
        return 0

    root = project_root()
    main_py = root / "main.py"
    hud_dir = root / "hud"
    package_json = hud_dir / "package.json"
    python_exe = find_python(root)

    if not main_py.exists():
        release_single_instance_lock(mutex)
        show_error(f"A.S.T.A. runtime not found:\n\n{main_py}")
        return 1

    if not python_exe.exists():
        release_single_instance_lock(mutex)
        show_error(
            "A.S.T.A. Python environment was not found.\n\n"
            "Expected .venv\\Scripts\\python.exe next to ASTA.exe."
        )
        return 1

    npm = find_npm()
    if not package_json.exists() or not npm:
        release_single_instance_lock(mutex)
        show_error("A.S.T.A. HUD dependencies were not found.\n\nInstall Node.js/npm and run npm install inside the hud directory.")
        return 1

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["ASTA_PRELAUNCHED_HUD"] = "1"

    hud_process = None
    python_process = None

    try:
        # Launch Electron immediately. The HUD assembles visually while the
        # Python/AI stack initializes in parallel.
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
            creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            shell=False,
        )

        return python_process.wait()
    except OSError as exc:
        show_error(f"Could not start A.S.T.A.\n\n{type(exc).__name__}: {exc}")
        return 1
    finally:
        terminate_process_tree(python_process)
        terminate_process_tree(hud_process)
        release_single_instance_lock(mutex)


if __name__ == "__main__":
    raise SystemExit(main())
