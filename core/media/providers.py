from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import platform
import re
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Protocol

import requests

from .request import MediaRequest


def _normalize_media_text(value: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
    )


@dataclass(frozen=True, slots=True)
class MediaResult:
    success: bool
    provider: str
    operation: str
    message: str
    query: str | None = None
    output: dict | None = None
    error: str | None = None


class MediaProvider(Protocol):
    name: str
    aliases: tuple[str, ...]
    priority: int

    def supports(self, request: MediaRequest) -> bool:
        ...

    def execute(self, request: MediaRequest) -> MediaResult:
        ...


class WindowsMediaProvider:
    name = "system"
    aliases = ("windows", "global", "media keys")
    application_name = None
    priority = 10

    _VIRTUAL_KEYS = {
        "play": 0xB3,
        "toggle": 0xB3,
        "pause": 0xB3,
        "next": 0xB0,
        "previous": 0xB1,
        "stop": 0xB2,
    }

    def supports(self, request: MediaRequest) -> bool:
        return (
            platform.system() == "Windows"
            and request.operation in self._VIRTUAL_KEYS
            and not request.query
        )

    def execute(self, request: MediaRequest) -> MediaResult:
        if not self.supports(request):
            return MediaResult(
                success=False,
                provider=self.name,
                operation=request.operation,
                query=request.query,
                message="",
                error="Windows media-key control does not support that request.",
            )

        vk = self._VIRTUAL_KEYS[request.operation]
        try:
            user32 = ctypes.windll.user32
            key_up = 0x0002
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, key_up, 0)
        except Exception as exc:
            return MediaResult(
                success=False,
                provider=self.name,
                operation=request.operation,
                query=request.query,
                message="",
                error=f"Media key control failed: {type(exc).__name__}: {exc}",
            )

        messages = {
            "play": "Resumed media playback.",
            "toggle": "Toggled media playback.",
            "pause": "Paused media playback.",
            "next": "Skipped to the next track.",
            "previous": "Returned to the previous track.",
            "stop": "Stopped media playback.",
        }
        return MediaResult(
            success=True,
            provider=self.name,
            operation=request.operation,
            query=request.query,
            message=messages[request.operation],
            output={"provider": self.name, "operation": request.operation},
        )


class SpotifyProvider:
    name = "spotify"
    aliases = ("spoti",)
    application_name = "Spotify"
    priority = 20
    developer_dashboard_url = "https://developer.spotify.com/dashboard"

    def __init__(self, notify_user=None):
        self._system = WindowsMediaProvider()
        self._notify_user = notify_user
        self._last_setup_open_at = 0.0
        self._setup_open_cooldown = 30.0
        self.client_id = os.getenv("ASTA_SPOTIFY_CLIENT_ID", "").strip()
        self.redirect_uri = os.getenv(
            "ASTA_SPOTIFY_REDIRECT_URI",
            "http://127.0.0.1:8765/callback",
        ).strip()
        self.token_path = os.getenv(
            "ASTA_SPOTIFY_TOKEN_PATH",
            "data/spotify_token.json",
        ).strip()

    @property
    def configured(self) -> bool:
        return bool(self.client_id)

    def _announce(self, text):
        callback = self._notify_user
        if callable(callback):
            try:
                callback(text)
            except Exception as exc:
                print(
                    f"[Media/Spotify] User notification failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _open_developer_setup(self):
        now = time.monotonic()
        if now - self._last_setup_open_at < self._setup_open_cooldown:
            return

        self._last_setup_open_at = now
        print(
            "[Media/Spotify] Spotify is not configured; opening the developer setup page.",
            flush=True,
        )
        webbrowser.open(
            self.developer_dashboard_url,
            new=2,
            autoraise=True,
        )

    def supports(self, request: MediaRequest) -> bool:
        if request.operation == "play" and request.query:
            return True
        return request.operation in {
            "play",
            "pause",
            "toggle",
            "next",
            "previous",
            "stop",
        } and not request.query

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return _normalize_media_text(value)

    def _load_token(self):
        try:
            with open(self.token_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def _save_token(self, token):
        path = os.path.abspath(self.token_path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = dict(token)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _pkce_values():
        verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(48)
        ).rstrip(b"=").decode("ascii")
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        state = secrets.token_urlsafe(24)
        return verifier, challenge, state

    def _authorize(self):
        if not self.client_id:
            raise RuntimeError(
                "Spotify integration is not configured. "
                "Set ASTA_SPOTIFY_CLIENT_ID first."
            )

        parsed = urllib.parse.urlparse(self.redirect_uri)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
            raise RuntimeError(
                "Spotify redirect URI must use a loopback address such as "
                "http://127.0.0.1:8765/callback."
            )

        bind_host = parsed.hostname
        bind_port = parsed.port or 8765
        callback_path = parsed.path or "/"

        verifier, challenge, state = self._pkce_values()
        callback = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.parse_qs(
                    urllib.parse.urlparse(self.path).query
                )
                callback["code"] = query.get("code", [None])[0]
                callback["state"] = query.get("state", [None])[0]
                callback["error"] = query.get("error", [None])[0]

                body = (
                    "A.S.T.A. Spotify authorization complete. "
                    "You can return to A.S.T.A."
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        server = HTTPServer((bind_host, bind_port), CallbackHandler)
        auth_params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
            "state": state,
            "scope": "user-read-playback-state user-modify-playback-state",
        }
        auth_url = (
            "https://accounts.spotify.com/authorize?"
            + urllib.parse.urlencode(auth_params)
        )

        self._announce(
            "Spotify isn't authorized yet. I'll open the Spotify authentication page."
        )
        print("[Media/Spotify] Opening authorization in the browser.", flush=True)
        webbrowser.open(auth_url, new=2, autoraise=True)

        deadline = time.monotonic() + 180.0
        try:
            while time.monotonic() < deadline and not callback.get("code"):
                server.timeout = min(1.0, max(0.05, deadline - time.monotonic()))
                server.handle_request()
        finally:
            server.server_close()

        if callback.get("error"):
            raise RuntimeError(
                f"Spotify authorization was denied: {callback['error']}."
            )
        if callback.get("state") != state:
            raise RuntimeError("Spotify authorization state validation failed.")
        if not callback.get("code"):
            raise RuntimeError("Spotify authorization timed out.")

        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "authorization_code",
                "code": callback["code"],
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "code_verifier": verifier,
            },
            timeout=15,
        )
        response.raise_for_status()
        token = response.json()
        token["expires_at"] = time.time() + int(token.get("expires_in", 3600))
        self._save_token(token)
        return token

    def _invalidate_access_token(self):
        token = self._load_token()
        if not token:
            return
        token = dict(token)
        token["access_token"] = ""
        token["expires_at"] = 0
        self._save_token(token)

    def _get_access_token(self, *, force_refresh=False):
        token = self._load_token()
        if token:
            expires_at = float(token.get("expires_at", 0) or 0)
            if not force_refresh and expires_at > time.time() + 60:
                return token.get("access_token")

            refresh_token = token.get("refresh_token")
            if refresh_token and self.client_id:
                response = requests.post(
                    "https://accounts.spotify.com/api/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": self.client_id,
                    },
                    timeout=15,
                )
                if response.ok:
                    refreshed = response.json()
                    merged = dict(token)
                    merged.update(refreshed)
                    merged["expires_at"] = (
                        time.time() + int(refreshed.get("expires_in", 3600))
                    )
                    if not merged.get("refresh_token"):
                        merged["refresh_token"] = refresh_token
                    self._save_token(merged)
                    return merged.get("access_token")

        return self._authorize().get("access_token")

    def _spotify_request(self, method, path, *, params=None, json_body=None):
        token = self._get_access_token()
        if not token:
            raise RuntimeError("Spotify authorization did not return an access token.")

        response = requests.request(
            method,
            f"https://api.spotify.com/v1{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            json=json_body,
            timeout=15,
        )

        if response.status_code == 401:
            self._invalidate_access_token()
            token = self._get_access_token(force_refresh=True)
            if not token:
                raise RuntimeError("Spotify authorization could not be refreshed.")
            response = requests.request(
                method,
                f"https://api.spotify.com/v1{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                json=json_body,
                timeout=15,
            )

        return response

    def _api(self, method, path, *, token, params=None, json_body=None):
        response = requests.request(
            method,
            f"https://api.spotify.com/v1{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            json=json_body,
            timeout=15,
        )

        if response.status_code == 401:
            # The caller can retry once after refreshing the token.
            raise PermissionError("Spotify authorization expired.")
        if response.status_code >= 400:
            detail = response.text.strip()
            raise RuntimeError(
                f"Spotify API error {response.status_code}: "
                f"{detail[:400]}"
            )

        if not response.content:
            return None
        return response.json()

    def _search_track(self, token, query):
        response = self._spotify_request(
            "GET",
            "/search",
            params={
                "q": query,
                "type": "track",
                "limit": 5,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Spotify search failed ({response.status_code}): "
                f"{response.text.strip()[:350]}"
            )
        payload = response.json()
        tracks = ((payload or {}).get("tracks") or {}).get("items") or []
        if not tracks:
            raise LookupError(f"No Spotify track matched '{query}'.")

        target = self._normalize_search_text(query)

        def score(track):
            title = self._normalize_search_text(track.get("name", ""))
            artists = " ".join(
                self._normalize_search_text(item.get("name", ""))
                for item in (track.get("artists") or [])
            )
            if title == target:
                return 100
            if target and target in title:
                return 90
            if title and title in target:
                return 80
            if target and target in artists:
                return 60
            overlap = len(set(target.split()) & set((title + " " + artists).split()))
            return overlap

        track = max(tracks, key=score)
        uri = track.get("uri")
        if not uri:
            raise LookupError(f"Spotify returned no playable URI for '{query}'.")

        artists = ", ".join(
            str(item.get("name", "")).strip()
            for item in (track.get("artists") or [])
            if item.get("name")
        )
        return {
            "uri": uri,
            "name": track.get("name") or query,
            "artists": artists,
        }

    def _select_device(self, token):
        state_response = self._spotify_request("GET", "/me/player")
        if state_response.status_code == 200:
            state = state_response.json()
            if state and state.get("device", {}).get("id"):
                return state["device"]["id"]

        devices_response = self._spotify_request(
            "GET",
            "/me/player/devices",
        )
        if devices_response.status_code >= 400:
            raise RuntimeError(
                f"Spotify device lookup failed ({devices_response.status_code}): "
                f"{devices_response.text.strip()[:350]}"
            )
        payload = devices_response.json() or {}
        devices = payload.get("devices") or []
        active = next(
            (device for device in devices if device.get("is_active") and device.get("id")),
            None,
        )
        if active:
            return active["id"]

        computer = next(
            (
                device
                for device in devices
                if device.get("type") == "Computer" and device.get("id")
            ),
            None,
        )
        return computer.get("id") if computer else None

    def _play_query(self, query):
        if not self.configured:
            self._open_developer_setup()
            raise RuntimeError(
                "Spotify is not configured yet. The Spotify Developer setup page was opened. "
                "Set ASTA_SPOTIFY_CLIENT_ID and restart A.S.T.A."
            )

        token = self._get_access_token()
        if not token:
            raise RuntimeError("Spotify authorization did not return an access token.")

        track = self._search_track(token, query)
        device_id = self._select_device(token)
        body = {"uris": [track["uri"]]}
        if device_id:
            body["device_id"] = device_id

        response = requests.put(
            "https://api.spotify.com/v1/me/player/play",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
            timeout=15,
        )
        if response.status_code in {401, 403, 404}:
            detail = response.text.strip()
            raise RuntimeError(
                f"Spotify playback failed ({response.status_code}). "
                f"{detail[:350]}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Spotify playback failed ({response.status_code}): "
                f"{response.text.strip()[:350]}"
            )

        return track

    def _api_control(self, operation):
        if not self.configured:
            # Basic transport controls can still work against the active
            # Windows media session without requiring Spotify credentials.
            return self._system.execute(
                MediaRequest(operation=operation)
            )

        token = self._get_access_token()
        if not token:
            raise RuntimeError("Spotify authorization did not return an access token.")

        endpoints = {
            "play": ("PUT", "/me/player/play"),
            "pause": ("PUT", "/me/player/pause"),
            "next": ("POST", "/me/player/next"),
            "previous": ("POST", "/me/player/previous"),
        }

        if operation == "stop":
            return self._system.execute(MediaRequest(operation="stop"))

        method, path = endpoints[operation]
        response = self._spotify_request(method, path)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Spotify playback control failed ({response.status_code}): "
                f"{response.text.strip()[:350]}"
            )
        messages = {
            "play": "Resumed Spotify playback.",
            "pause": "Paused Spotify playback.",
            "next": "Skipped to the next Spotify track.",
            "previous": "Returned to the previous Spotify track.",
        }
        return MediaResult(
            success=True,
            provider=self.name,
            operation=operation,
            message=messages[operation],
            output={"provider": self.name, "operation": operation},
        )

    def execute(self, request: MediaRequest) -> MediaResult:
        if request.operation == "play" and request.query:
            try:
                track = self._play_query(request.query)
            except PermissionError:
                try:
                    track = self._play_query(request.query)
                except Exception as exc:
                    return MediaResult(
                        success=False,
                        provider=self.name,
                        operation=request.operation,
                        query=request.query,
                        message="",
                        error=str(exc),
                    )
            except Exception as exc:
                return MediaResult(
                    success=False,
                    provider=self.name,
                    operation=request.operation,
                    query=request.query,
                    message="",
                    error=str(exc),
                )

            return MediaResult(
                success=True,
                provider=self.name,
                operation=request.operation,
                query=request.query,
                message=(
                    f"Playing {track['name']}"
                    + (f" by {track['artists']}" if track["artists"] else "")
                    + "."
                ),
                output={
                    "provider": self.name,
                    "operation": request.operation,
                    "query": request.query,
                    "track": track["name"],
                    "artists": track["artists"],
                    "uri": track["uri"],
                    "playback_started": True,
                },
            )

        if request.operation in {
            "play",
            "pause",
            "toggle",
            "next",
            "previous",
            "stop",
        }:
            operation = "play" if request.operation == "toggle" else request.operation
            try:
                result = self._api_control(operation)
                if result.provider == self._system.name:
                    return MediaResult(
                        success=result.success,
                        provider=self.name,
                        operation=request.operation,
                        query=request.query,
                        message=(
                            "Toggled Spotify playback."
                            if request.operation == "toggle"
                            else result.message
                        ),
                        output=result.output,
                        error=result.error,
                    )
                return result
            except Exception as exc:
                return MediaResult(
                    success=False,
                    provider=self.name,
                    operation=request.operation,
                    query=request.query,
                    message="",
                    error=str(exc),
                )

        return MediaResult(
            success=False,
            provider=self.name,
            operation=request.operation,
            query=request.query,
            message="",
            error="Unsupported Spotify media operation.",
        )


class MediaManager:
    def __init__(self, providers=None):
        self._providers: list[MediaProvider] = list(
            providers or (
                SpotifyProvider(),
                WindowsMediaProvider(),
            )
        )
        self._by_name: dict[str, MediaProvider] = {}
        for provider in self._providers:
            for name in (provider.name, *provider.aliases):
                self._by_name[str(name).strip().lower()] = provider

    def providers(self) -> tuple[str, ...]:
        return tuple(provider.name for provider in self._providers)

    def provider_for_application(self, application_name: str):
        normalized = _normalize_media_text(application_name)
        if not normalized:
            return None

        tokens = set(normalized.split())
        for provider in self._providers:
            aliases = {
                _normalize_media_text(provider.name),
                *(
                    _normalize_media_text(alias)
                    for alias in getattr(provider, "aliases", ())
                ),
            }
            for alias in aliases:
                if alias and (
                    alias == normalized
                    or alias in tokens
                    or alias in normalized
                ):
                    return provider.name
        return None

    def application_for_provider(self, provider_name: str):
        normalized = _normalize_media_text(provider_name)
        if not normalized:
            return None

        for provider in self._providers:
            names = {
                _normalize_media_text(provider.name),
                *(
                    _normalize_media_text(alias)
                    for alias in getattr(provider, "aliases", ())
                ),
            }
            if normalized in names:
                return getattr(provider, "application_name", None)
        return None

    def execute(self, request: MediaRequest) -> MediaResult:
        explicit = str(request.provider or "").strip().lower()
        if not explicit and request.query:
            explicit = os.getenv("ASTA_MEDIA_DEFAULT_PROVIDER", "").strip().lower()
        if not explicit and request.query:
            return MediaResult(
                success=False,
                provider="",
                operation=request.operation,
                query=request.query,
                message="",
                error=(
                    "A media provider is required for catalog playback. "
                    "Specify one or set ASTA_MEDIA_DEFAULT_PROVIDER."
                ),
            )
        if explicit:
            provider = self._by_name.get(explicit)
            if provider is None:
                return MediaResult(
                    success=False,
                    provider=explicit,
                    operation=request.operation,
                    query=request.query,
                    message="",
                    error=f"Unknown media provider '{request.provider}'.",
                )
            if not provider.supports(request):
                return MediaResult(
                    success=False,
                    provider=provider.name,
                    operation=request.operation,
                    query=request.query,
                    message="",
                    error=(
                        f"Media provider '{provider.name}' does not support that request."
                    ),
                )
            return provider.execute(request)

        candidates = [
            provider
            for provider in self._providers
            if provider.supports(request)
        ]
        if not candidates:
            return MediaResult(
                success=False,
                provider="",
                operation=request.operation,
                query=request.query,
                message="",
                error="No media provider can handle that request.",
            )

        provider = sorted(
            candidates,
            key=lambda item: (-int(getattr(item, "priority", 0)), item.name),
        )[0]
        return provider.execute(request)
