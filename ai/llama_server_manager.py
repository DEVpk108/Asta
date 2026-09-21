from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


class LlamaServerManager:
    """Own the local llama-server process when A.S.T.A. starts it."""

    def __init__(
        self,
        *,
        base_url: str,
        server_path: str | None = None,
        model_path: str | None = None,
        startup_timeout: float | None = None,
        context_size: int | None = None,
        gpu_layers: int | None = None,
        jinja: bool | None = None,
        reasoning: str | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.server_path = (
            server_path
            or os.getenv("ASTA_LLAMA_SERVER_PATH")
            or self._default_server_path()
        )
        self.model_path = (
            model_path
            or os.getenv("ASTA_LLM_MODEL_PATH")
            or self._model_path_from_legacy_env()
        )
        self.startup_timeout = float(
            startup_timeout
            if startup_timeout is not None
            else os.getenv("ASTA_LLAMA_SERVER_STARTUP_TIMEOUT", "120")
        )
        self.context_size = int(
            context_size
            if context_size is not None
            else os.getenv("ASTA_LLM_CONTEXT_SIZE", "8192")
        )
        self.gpu_layers = int(
            gpu_layers
            if gpu_layers is not None
            else os.getenv("ASTA_LLM_GPU_LAYERS", "99")
        )
        self.jinja = self._env_bool(
            "ASTA_LLM_JINJA",
            True if jinja is None else jinja,
        )
        self.reasoning = (
            reasoning
            if reasoning is not None
            else os.getenv("ASTA_LLM_REASONING", "off")
        ).strip()

        self.process: subprocess.Popen | None = None
        self.owned = False

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _default_server_path() -> str | None:
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"

        candidates = [
            shutil.which(executable),
            shutil.which("llama-server"),
            Path.cwd() / executable,
            Path(__file__).resolve().parents[2] / "llama" / executable,
        ]

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists() and path.is_file():
                return str(path)

        return None

    @staticmethod
    def _model_path_from_legacy_env() -> str | None:
        value = os.getenv("ASTA_LLM_MODEL")
        if not value:
            return None
        candidate = Path(value).expanduser()
        return str(candidate) if candidate.exists() else None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure_running(self) -> bool:
        if self._server_is_ready():
            print(
                "[AI] llama.cpp server already running; A.S.T.A. will not own it.",
                flush=True,
            )
            return True

        if not self._autostart_enabled():
            print(
                "[AI] llama.cpp auto-start disabled "
                "(ASTA_LLM_AUTOSTART=0).",
                flush=True,
            )
            return False

        if not self._is_loopback_host():
            print(
                "[AI] llama.cpp auto-start skipped for non-loopback "
                "ASTA_LLM_BASE_URL.",
                flush=True,
            )
            return False

        server = self._resolve_server_path()
        model = self._resolve_model_path(server)
        if not server or not model:
            print(
                "[AI] llama.cpp auto-start unavailable: set "
                "ASTA_LLAMA_SERVER_PATH and ASTA_LLM_MODEL_PATH.",
                flush=True,
            )
            return False

        command = self._build_command(server, model)
        print(
            "[AI] Starting llama.cpp server: "
            f"{Path(server).name} -> {Path(model).name}",
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
                f"[AI] Failed to start llama.cpp server: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

        if self._wait_until_ready():
            print(
                f"[AI] llama.cpp server ready (PID {self.process.pid}).",
                flush=True,
            )
            return True

        print(
            "[AI] llama.cpp server did not become ready before the startup timeout.",
            flush=True,
        )
        self.stop()
        return False

    def stop(self) -> None:
        process = self.process
        owned = self.owned

        self.process = None
        self.owned = False

        if not owned or process is None:
            return

        if process.poll() is not None:
            return

        print(
            f"[AI] Stopping llama.cpp server (PID {process.pid})...",
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
                        subprocess,
                        "CREATE_NO_WINDOW",
                        0x08000000,
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

    def _autostart_enabled(self) -> bool:
        return self._env_bool("ASTA_LLM_AUTOSTART", True)

    def _resolve_server_path(self) -> str | None:
        value = str(self.server_path).strip() if self.server_path else ""
        if not value:
            return None

        path = Path(value).expanduser()
        if path.exists() and path.is_file():
            return str(path)

        return shutil.which(value)

    def _resolve_model_path(self, server_path: str | None) -> str | None:
        value = str(self.model_path).strip() if self.model_path else ""
        if value:
            path = Path(value).expanduser()
            if path.exists() and path.is_file():
                return str(path)
            print(
                f"[AI] Configured llama.cpp model does not exist: {path}",
                flush=True,
            )
            return None

        if not server_path:
            return None

        server_dir = Path(server_path).resolve().parent
        model_dir = server_dir / "models"
        candidates = sorted(model_dir.glob("*.gguf"))
        if len(candidates) == 1:
            return str(candidates[0])

        if len(candidates) > 1:
            print(
                "[AI] Multiple GGUF files found under "
                f"{model_dir}; set ASTA_LLM_MODEL_PATH explicitly.",
                flush=True,
            )

        return None

    def _build_command(self, server_path: str, model_path: str) -> list[str]:
        host, port = self._server_endpoint()
        command = [
            server_path,
            "-m",
            model_path,
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

        if self.reasoning:
            command.extend(["--reasoning", self.reasoning])

        return command

    def _server_endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        return host, port

    def _is_loopback_host(self) -> bool:
        host, _ = self._server_endpoint()
        return host in {"127.0.0.1", "localhost", "::1"}

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
