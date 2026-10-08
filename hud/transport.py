"""Local A.S.T.A. HUD transport.

V1 uses a localhost TCP connection carrying one JSON object per line.
The transport is intentionally dependency-free so the Python runtime can
communicate with the Electron HUD without coupling the Kernel to Electron.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import socket
import threading
import time
from dataclasses import asdict
from pathlib import PureWindowsPath
from typing import Any, Callable
from urllib.parse import urlsplit


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765
TOKEN_ENV = "ASTA_HUD_TOKEN"

# A client must authenticate within this window or it is disconnected.
_AUTH_TIMEOUT_SECONDS = 5.0
# Reject clients that stream data without line breaks (memory exhaustion).
_MAX_BUFFER_BYTES = 256 * 1024
_SECRET_FILE_PART = re.compile(
    r"(?:^|[._-])(?:secrets?|credentials?|tokens?)(?:[._-]|$)",
    re.IGNORECASE,
)
_PRIVATE_PATH_PARTS = {
    ".git",
    ".ssh",
    ".aws",
    ".azure",
    ".venv",
    "venv",
    "node_modules",
    "credentials",
    "secrets",
}


class HUDTransport:
    """Small localhost JSON-lines server for HUD state and live telemetry."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        token: str | None = None,
    ):
        self.host = host or os.getenv("ASTA_HUD_HOST", DEFAULT_HOST)
        self.port = int(port or os.getenv("ASTA_HUD_PORT", DEFAULT_PORT))
        # Every client must present this shared secret in a ``hud.hello``
        # message before it receives state or may send commands. Without it,
        # any local process -- or a web page issuing a cross-protocol HTTP
        # request to 127.0.0.1 -- could inject commands into A.S.T.A.
        self.token = token if token is not None else os.getenv(TOKEN_ENV, "")
        self._authenticated: set[socket.socket] = set()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_state_message: dict[str, Any] | None = None
        self._last_workspace_message: dict[str, Any] | None = None
        self._last_audio_message: dict[str, Any] | None = None
        self._last_lifecycle_message: dict[str, Any] | None = None
        self._last_chat_history_message: dict[str, Any] | None = None
        self._last_chat_sessions_message: dict[str, Any] | None = None
        self._command_handler: Callable[[dict[str, Any]], None] | None = None

    def set_command_handler(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        self._command_handler = handler

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(4)
        server.settimeout(0.5)
        self._server = server
        self._thread = threading.Thread(target=self._accept_loop, name="HUDTransport", daemon=True)
        self._thread.start()
        print(f"[HUD] Transport listening on {self.host}:{self.port}", flush=True)

    def publish_state(self, state) -> None:
        message = {"type": "hud.state", "version": 1, "state": asdict(state)}
        self._last_state_message = message
        self._broadcast(message)

    def publish_workspace_context(self, workspace: dict[str, Any] | None) -> None:
        """Publish only display-safe project identity, never the full workspace."""
        value = workspace if isinstance(workspace, dict) else {}
        recent_files = self._safe_recent_files(value.get("recent_files"))
        display = {
            "project_name": self._safe_text(value.get("project_name"), 96),
            "repository": self._repository_label(value.get("repository")),
            "branch": self._safe_text(value.get("branch"), 72),
            "recent_files": recent_files,
        }
        message = {"type": "hud.workspace", "version": 1, "workspace": display}
        self._last_workspace_message = message
        self._broadcast(message)

    def publish_audio_level(self, level: float) -> None:
        try:
            value = float(level)
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(1.0, value))
        message = {"type": "hud.audio", "version": 1, "audio": {"level": value}}
        self._last_audio_message = message
        self._broadcast(message)

    def publish_chat(self, *, role: str, text: str) -> None:
        normalized_role = "user" if role == "user" else "assistant"
        value = str(text or "").strip()
        if not value:
            return
        message = {
            "type": "hud.chat",
            "version": 1,
            "chat": {"role": normalized_role, "text": value},
        }
        history = self._last_chat_history_message
        if history is None:
            history = {
                "type": "hud.chat_history",
                "version": 1,
                "session_id": None,
                "sessions": [],
                "messages": [],
            }
            self._last_chat_history_message = history
        history["messages"].append({"role": normalized_role, "text": value})
        if len(history["messages"]) > 200:
            history["messages"] = history["messages"][-200:]
        self._broadcast(message)

    def publish_chat_history(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        sessions: list[dict[str, Any]] | None = None,
    ) -> None:
        safe_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = "user" if message.get("role") == "user" else "assistant"
            text = str(message.get("text") or "").strip()
            if text:
                safe_messages.append({"role": role, "text": text})
        safe_sessions = [dict(session) for session in (sessions or []) if isinstance(session, dict)]
        self._last_chat_history_message = {
            "type": "hud.chat_history",
            "version": 1,
            "session_id": session_id,
            "sessions": safe_sessions,
            "messages": safe_messages[-200:],
        }
        self._broadcast(self._last_chat_history_message)

    def publish_chat_sessions(self, sessions: list[dict[str, Any]]) -> None:
        safe_sessions = [dict(session) for session in sessions if isinstance(session, dict)]
        self._last_chat_sessions_message = {
            "type": "hud.chat_sessions",
            "version": 1,
            "sessions": safe_sessions,
        }
        self._broadcast(self._last_chat_sessions_message)

    def publish_window(self, action: str) -> None:
        """Ask the HUD window to minimize/restore (not cached for replay)."""
        self._broadcast({"type": "hud.window", "version": 1, "action": str(action)})

    def publish_lifecycle(self, status: str) -> None:
        message = {"type": "hud.lifecycle", "version": 1, "lifecycle": {"status": str(status)}}
        self._last_lifecycle_message = message
        self._broadcast(message)

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
            self._authenticated.clear()
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def _accept_loop(self) -> None:
        server = self._server
        if server is None:
            return
        while not self._stop.is_set():
            try:
                client, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(0.5)
            threading.Thread(target=self._client_loop, args=(client,), name="HUDTransportClient", daemon=True).start()

    def _register_authenticated_client(self, client: socket.socket) -> None:
        with self._clients_lock:
            self._authenticated.add(client)
            self._clients.add(client)
        for cached in (
            self._last_state_message,
            self._last_workspace_message,
            self._last_audio_message,
            self._last_lifecycle_message,
            self._last_chat_history_message,
            self._last_chat_sessions_message,
        ):
            if cached is not None:
                self._send_to_client(client, cached)

    @staticmethod
    def _safe_text(value: object, limit: int) -> str:
        if not isinstance(value, str):
            return ""
        text = "".join(
            character for character in value
            if character.isprintable()
        )
        return text.strip()[:limit]

    @classmethod
    def _repository_label(cls, value: object) -> str:
        """Return a non-clickable host/path label with credentials removed."""
        raw = cls._safe_text(value, 2048)
        if not raw:
            return ""
        try:
            parsed = urlsplit(raw)
        except ValueError:
            return ""

        if parsed.scheme and parsed.netloc:
            if parsed.scheme.lower() not in {"http", "https", "ssh", "git"}:
                return ""
            host = parsed.hostname or ""
            path = parsed.path
        elif ":" in raw:
            host_part, path = raw.split(":", 1)
            host = host_part.rsplit("@", 1)[-1]
        else:
            return ""

        if not host or not re.fullmatch(r"[A-Za-z0-9.-]+", host):
            return ""
        path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
        parts = [part for part in path.split("/") if part]
        if not parts or any(part == ".." for part in parts):
            return ""
        label = "/".join(parts)
        if label.lower().endswith(".git"):
            label = label[:-4]
        return f"{host}/{label}"[:180]

    @staticmethod
    def _safe_recent_files(value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        files: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                continue
            normalized = raw.strip().replace("\\", "/")
            windows = PureWindowsPath(raw.strip())
            if (
                not normalized
                or "\x00" in normalized
                or normalized.startswith("/")
                or windows.is_absolute()
                or windows.drive
            ):
                continue
            parts = [part for part in normalized.split("/") if part not in {"", "."}]
            if (
                not parts
                or any(part == ".." for part in parts)
                or any(not part.isprintable() for part in parts)
            ):
                continue
            lower_parts = [part.lower() for part in parts]
            if any(part in _PRIVATE_PATH_PARTS for part in lower_parts):
                continue
            if any(
                part == ".env"
                or (part.startswith(".env.") and part != ".env.example")
                or PureWindowsPath(part).suffix.lower() in {".key", ".pem", ".p12", ".pfx"}
                or _SECRET_FILE_PART.search(part)
                for part in lower_parts
            ):
                continue
            path = "/".join(parts)
            if len(path) > 160 or path in files:
                continue
            files.append(path)
            if len(files) >= 4:
                break
        return files

    def _is_valid_hello(self, message: Any) -> bool:
        if not isinstance(message, dict) or message.get("type") != "hud.hello":
            return False
        if not self.token:
            # No shared secret configured (tests / manual development).
            return True
        supplied = message.get("token")
        if not isinstance(supplied, str):
            return False
        return hmac.compare_digest(supplied.encode("utf-8"), self.token.encode("utf-8"))

    def _client_loop(self, client: socket.socket) -> None:
        buffer = b""
        authenticated = False
        deadline = time.monotonic() + _AUTH_TIMEOUT_SECONDS
        try:
            while not self._stop.is_set():
                if not authenticated and time.monotonic() > deadline:
                    break
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                buffer += data
                if len(buffer) > _MAX_BUFFER_BYTES:
                    break
                lines = buffer.split(b"\n")
                buffer = lines.pop() or b""
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        if not authenticated:
                            # Not a HUD client (e.g. an HTTP request). Drop it
                            # instead of skipping lines until a JSON body.
                            return
                        continue
                    if not authenticated:
                        if not self._is_valid_hello(message):
                            print("[HUD] Rejected unauthenticated transport client.", flush=True)
                            return
                        authenticated = True
                        self._register_authenticated_client(client)
                        continue
                    self._handle_incoming(message)
        finally:
            self._remove_client(client)

    def _handle_incoming(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict) or self._command_handler is None:
            return
        try:
            self._command_handler(message)
        except Exception as exc:
            print(f"[HUD] Input handler failed: {type(exc).__name__}: {exc}", flush=True)

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            if not self._send_to_client(client, message):
                self._remove_client(client)

    @staticmethod
    def _send_to_client(client: socket.socket, message: dict[str, Any]) -> bool:
        payload = (json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        try:
            client.sendall(payload)
            return True
        except OSError:
            return False

    def _remove_client(self, client: socket.socket) -> None:
        with self._clients_lock:
            self._clients.discard(client)
            self._authenticated.discard(client)
        try:
            client.close()
        except OSError:
            pass
