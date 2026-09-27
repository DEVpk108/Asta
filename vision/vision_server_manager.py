from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


class VisionServerManager:
    """Lazily own a local llama-server process for the vision model.

    A.S.T.A. only starts the server on the first semantic vision request unless
    ASTA_VISION_PRELOAD=1 is enabled. An already-running loopback server is
    reused and never owned or terminated by A.S.T.A.
    """

    DEFAULT_LLAMA_ROOT = Path(r"E:\Projects\llama")
    DEFAULT_MODEL_DIR = DEFAULT_LLAMA_ROOT / "models"

    def __init__(
        self,
        *,
        base_url: str,
        server_path: str | None = None,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        startup_timeout: float | None = None,
        context_size: int | None = None,
        gpu_layers: int | None = None,
        jinja: bool | None = None,
        mmproj_offload: bool | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.server_path = (
            server_path
            or os.getenv("ASTA_VISION_SERVER_PATH")
            or os.getenv("ASTA_LLAMA_SERVER_PATH")
            or self._default_server_path()
        )
        self.model_path = (
            model_path
            or os.getenv("ASTA_VISION_MODEL_PATH")
        )
        self.mmproj_path = (
            mmproj_path
            or os.getenv("ASTA_VISION_MMPROJ_PATH")
        )
        self.startup_timeout = float(
            startup_timeout
            if startup_timeout is not None
            else os.getenv("ASTA_VISION_SERVER_STARTUP_TIMEOUT", "90")
        )
        self.context_size = int(
            context_size
            if context_size is not None
            else os.getenv("ASTA_VISION_CONTEXT_SIZE", "8192")
        )
        self.gpu_layers = int(
            gpu_layers
            if gpu_layers is not None
            else os.getenv("ASTA_VISION_GPU_LAYERS", "99")
        )
        self.jinja = self._env_bool(
            "ASTA_VISION_JINJA",
            True if jinja is None else jinja,
        )
        self.mmproj_offload = self._env_bool(
            "ASTA_VISION_MMPROJ_OFFLOAD",
            True if mmproj_offload is None else mmproj_offload,
        )
        self.preload = self._env_bool("ASTA_VISION_PRELOAD", False)

        self.process: subprocess.Popen | None = None
        self.owned = False

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _default_server_path(cls) -> str | None:
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates = [
            shutil.which(executable),
            shutil.which("llama-server"),
            cls.DEFAULT_LLAMA_ROOT / executable,
            Path.cwd() / executable,
            Path(__file__).resolve().parents[1] / "llama" / executable,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists() and path.is_file():
                return str(path)
        return None

    @classmethod
    def default_model_dir(cls) -> Path:
        value = os.getenv("ASTA_VISION_MODEL_DIR")
        return (
            Path(value).expanduser()
            if value
            else cls.DEFAULT_MODEL_DIR
        )

    @staticmethod
    def _is_mmproj(path: Path) -> bool:
        return path.name.lower().startswith("mmproj")

    @classmethod
    def discover_model_paths(
        cls,
        model_dir: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        directory = model_dir or cls.default_model_dir()
        if not directory.exists():
            return None, None

        ggufs = sorted(
            path for path in directory.glob("*.gguf")
            if path.is_file()
        )
        mmprojs = [path for path in ggufs if cls._is_mmproj(path)]
        models = [path for path in ggufs if not cls._is_mmproj(path)]

        # Prefer LFM2.5-VL files when multiple GGUFs live in the shared folder.
        lfm_models = [
            path for path in models
            if "lfm2.5-vl" in path.name.lower()
            or "lfm2_5-vl" in path.name.lower()
            or "lfm2.5vl" in path.name.lower()
        ]
        lfm_mmprojs = [
            path for path in mmprojs
            if "lfm2.5-vl" in path.name.lower()
            or "lfm2_5-vl" in path.name.lower()
            or "lfm2.5vl" in path.name.lower()
        ]

        selected_model = (
            lfm_models[0] if len(lfm_models) == 1
            else models[0] if len(models) == 1
            else None
        )
        selected_mmproj = (
            lfm_mmprojs[0] if len(lfm_mmprojs) == 1
            else mmprojs[0] if len(mmprojs) == 1
            else None
        )
        return selected_model, selected_mmproj

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure_running(self) -> bool:
        if self._server_is_ready():
            return True

        server = self._resolve_server_path()
        model, mmproj = self._resolve_model_paths()
        if not server or not model or not mmproj:
            print(
                "[Vision] llama.cpp auto-start unavailable. Set "
                "ASTA_VISION_SERVER_PATH, ASTA_VISION_MODEL_PATH and "
                "ASTA_VISION_MMPROJ_PATH when auto-discovery cannot identify them.",
                flush=True,
            )
            return False

        command = self._build_command(server, model, mmproj)
        print(
            "[Vision] Starting llama.cpp vision server: "
            f"{Path(model).name} + {Path(mmproj).name}",
            flush=True,
        )

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                )
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                shell=False,
            )
            self.owned = True
        except OSError as exc:
            self.process = None
            self.owned = False
            print(
                f"[Vision] Failed to start llama-server: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

        if self._wait_until_ready():
            print(
                f"[Vision] llama.cpp vision server ready (PID {self.process.pid}).",
                flush=True,
            )
            return True

        print(
            "[Vision] Vision server did not become ready before the startup timeout.",
            flush=True,
        )
        self.stop()
        return False

    def warmup(self) -> bool:
        return self.ensure_running()

    def stop(self) -> None:
        process = self.process
        owned = self.owned
        self.process = None
        self.owned = False

        if not owned or process is None or process.poll() is not None:
            return

        print(
            f"[Vision] Stopping llama.cpp vision server (PID {process.pid})...",
            flush=True,
        )
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(
                        subprocess, "CREATE_NO_WINDOW", 0x08000000
                    ),
                )
            else:
                process.terminate()
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def _resolve_server_path(self) -> str | None:
        value = str(self.server_path).strip() if self.server_path else ""
        if not value:
            return None
        path = Path(value).expanduser()
        if path.exists() and path.is_file():
            return str(path)
        return shutil.which(value)

    def _resolve_model_paths(self) -> tuple[str | None, str | None]:
        model = (
            Path(self.model_path).expanduser()
            if self.model_path else None
        )
        mmproj = (
            Path(self.mmproj_path).expanduser()
            if self.mmproj_path else None
        )

        if model is None or not model.is_file():
            discovered_model, discovered_mmproj = self.discover_model_paths()
            if model is None:
                model = discovered_model
            if mmproj is None:
                mmproj = discovered_mmproj

        if mmproj is None or not mmproj.is_file():
            _, discovered_mmproj = self.discover_model_paths()
            if mmproj is None:
                mmproj = discovered_mmproj

        return (
            str(model) if model is not None and model.is_file() else None,
            str(mmproj) if mmproj is not None and mmproj.is_file() else None,
        )

    def _build_command(
        self,
        server_path: str,
        model_path: str,
        mmproj_path: str,
    ) -> list[str]:
        host, port = self._server_endpoint()
        command = [
            server_path,
            "-m",
            model_path,
            "--mmproj",
            mmproj_path,
            "--host",
            host,
            "--port",
            str(port),
            "-c",
            str(self.context_size),
            "-ngl",
            str(self.gpu_layers),
        ]
        if self.jinja:
            command.append("--jinja")
        if self.mmproj_offload:
            command.append("--mmproj-offload")
        else:
            command.append("--no-mmproj-offload")
        return command

    def _server_endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        return parsed.hostname or "127.0.0.1", parsed.port or 8090

    def _server_is_ready(self) -> bool:
        try:
            import requests

            with requests.Session() as session:
                session.trust_env = False
                response = session.get(
                    f"{self.base_url}/models",
                    timeout=1.5,
                )
                response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def _wait_until_ready(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return False
            if self._server_is_ready():
                return True
            time.sleep(0.25)
        return False
