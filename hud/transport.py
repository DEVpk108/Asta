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
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18765


class HUDTransport:
    """Small localhost JSON-lines server for HUD state messages."""

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or os.getenv("ASTA_HUD_HOST", DEFAULT_HOST)
        self.port = int(port or os.getenv("ASTA_HUD_PORT", DEFAULT_PORT))

        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_message: dict[str, Any] | None = None

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
        self._last_message = message
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

            if self._last_message is not None:
                self._send_to_client(client, self._last_message)

            threading.Thread(
                target=self._client_loop,
                args=(client,),
                name="HUDTransportClient",
                daemon=True,
            ).start()

    def _client_loop(self, client: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                # V1 is one-way. We keep the read side alive so the protocol
                # can grow to bidirectional commands later without changing
                # the connection model.
                for _line in data.splitlines():
                    if not _line:
                        continue
                    try:
                        json.loads(_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
        finally:
            self._remove_client(client)

    def _broadcast(self, message: dict[str, Any]) -> None:
        with self._clients_lock:
            clients = list(self._clients)

        for client in clients:
            if not self._send_to_client(client, message):
                self._remove_client(client)

    @staticmethod
    def _send_to_client(client: socket.socket, message: dict[str, Any]) -> bool:
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
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
