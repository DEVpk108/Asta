"""Local A.S.T.A. HUD transport.

V1 uses a localhost TCP connection carrying one JSON object per line.
The transport is intentionally dependency-free so the Python runtime can
communicate with the Electron HUD without coupling the Kernel to Electron.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import asdict
from typing import Any, Callable


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765


class HUDTransport:
    """Small localhost JSON-lines server for HUD state and live telemetry."""

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or os.getenv("ASTA_HUD_HOST", DEFAULT_HOST)
        self.port = int(port or os.getenv("ASTA_HUD_PORT", DEFAULT_PORT))

        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_state_message: dict[str, Any] | None = None
        self._last_audio_message: dict[str, Any] | None = None
        self._last_lifecycle_message: dict[str, Any] | None = None
        self._last_chat_history_message: dict[str, Any] | None = None
        self._last_chat_sessions_message: dict[str, Any] | None = None
        self._command_handler: Callable[[dict[str, Any]], None] | None = None

    def set_command_handler(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        """Register a callback for messages sent from the Electron HUD."""
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

        self._thread = threading.Thread(
            target=self._accept_loop,
            name="HUDTransport",
            daemon=True,
        )
        self._thread.start()

        print(
            f"[HUD] Transport listening on {self.host}:{self.port}",
            flush=True,
        )

    def publish_state(self, state) -> None:
        message = {
            "type": "hud.state",
            "version": 1,
            "state": asdict(state),
        }
        self._last_state_message = message
        self._broadcast(message)

    def publish_audio_level(self, level: float) -> None:
        """Publish a normalized 0..1 speech playback envelope."""
        try:
            value = float(level)
        except (TypeError, ValueError):
            value = 0.0

        value = max(0.0, min(1.0, value))
        message = {
            "type": "hud.audio",
            "version": 1,
            "audio": {
                "level": value,
            },
        }
        self._last_audio_message = message
        self._broadcast(message)

    def publish_chat(self, *, role: str, text: str) -> None:
        """Publish a conversation message and keep reconnect history current."""
        normalized_role = "user" if role == "user" else "assistant"
        value = str(text or "").strip()
        if not value:
            return

        message = {
            "type": "hud.chat",
            "version": 1,
            "chat": {
                "role": normalized_role,
                "text": value,
            },
        }

        history = self._last_chat_history_message
        if history is None:
            history = {
                "type": "hud.chat_history",
                "version": 1,
                "session_id": None,
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
    ) -> None:
        """Cache and publish the persisted history used when the HUD connects."""
        safe_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = "user" if message.get("role") == "user" else "assistant"
            text = str(message.get("text") or "").strip()
            if text:
                safe_messages.append({"role": role, "text": text})

        self._last_chat_history_message = {
            "type": "hud.chat_history",
            "version": 1,
            "session_id": session_id,
            "messages": safe_messages[-200:],
        }
        self._broadcast(self._last_chat_history_message)

    def publish_chat_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Publish the recent conversation/session index for the HUD sidebar."""
        self._last_chat_sessions_message = {
            "type": "hud.chat_sessions",
            "version": 1,
            "sessions": [dict(session) for session in sessions if isinstance(session, dict)],
        }
        self._broadcast(self._last_chat_sessions_message)

    def publish_lifecycle(self, status: str) -> None:
        """Publish a runtime lifecycle marker such as starting or ready."""
        message = {
            "type": "hud.lifecycle",
            "version": 1,
            "lifecycle": {
                "status": str(status),
            },
        }
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
            with self._clients_lock:
                self._clients.add(client)

            if self._last_state_message is not None:
                self._send_to_client(client, self._last_state_message)
            if self._last_audio_message is not None:
                self._send_to_client(client, self._last_audio_message)
            if self._last_lifecycle_message is not None:
                self._send_to_client(client, self._last_lifecycle_message)
            if self._last_chat_history_message is not None:
                self._send_to_client(client, self._last_chat_history_message)
            if self._last_chat_sessions_message is not None:
                self._send_to_client(client, self._last_chat_sessions_message)

            threading.Thread(
                target=self._client_loop,
                args=(client,),
                name="HUDTransportClient",
                daemon=True,
            ).start()

    def _client_loop(self, client: socket.socket) -> None:
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break

                buffer += data
                lines = buffer.split(b"\n")
                buffer = lines.pop() or b""

                for line in lines:
                    if not line:
                        continue
                    try:
                        message = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
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
            print(
                f"[HUD] Input handler failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

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
        try:
            client.close()
        except OSError:
            pass
