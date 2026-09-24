from __future__ import annotations

import ctypes
import os
import platform
import urllib.parse
import webbrowser
from dataclasses import dataclass
from typing import Protocol

from .request import MediaRequest


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
    priority = 20

    def __init__(self):
        self._system = WindowsMediaProvider()

    def supports(self, request: MediaRequest) -> bool:
        return (
            request.operation in {"play", "pause", "toggle", "next", "previous", "stop"}
            and (bool(request.query) or request.operation != "play" or platform.system() == "Windows")
        )

    @staticmethod
    def _open_search(query: str) -> None:
        encoded = urllib.parse.quote(query, safe="")
        uri = f"spotify:search:{encoded}"
        if platform.system() == "Windows":
            try:
                os.startfile(uri)
                return
            except OSError:
                pass
        webbrowser.open(
            f"https://open.spotify.com/search/{encoded}",
            new=0,
            autoraise=True,
        )

    def execute(self, request: MediaRequest) -> MediaResult:
        if request.query and request.operation == "play":
            try:
                self._open_search(request.query)
            except Exception as exc:
                return MediaResult(
                    success=False,
                    provider=self.name,
                    operation=request.operation,
                    query=request.query,
                    message="",
                    error=(
                        "Spotify search could not be opened: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            return MediaResult(
                success=True,
                provider=self.name,
                operation=request.operation,
                query=request.query,
                message=f"Opened Spotify search for {request.query}.",
                output={
                    "provider": self.name,
                    "operation": request.operation,
                    "query": request.query,
                    "playback_started": False,
                },
            )

        if request.operation in {"pause", "toggle", "next", "previous", "stop"}:
            return self._system.execute(request)

        if request.operation == "play":
            result = self._system.execute(request)
            if result.success:
                return MediaResult(
                    success=True,
                    provider=self.name,
                    operation=request.operation,
                    query=request.query,
                    message="Resumed Spotify playback.",
                    output={
                        "provider": self.name,
                        "operation": request.operation,
                        "playback_started": True,
                    },
                )
            return result

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

    def execute(self, request: MediaRequest) -> MediaResult:
        explicit = str(request.provider or "").strip().lower()
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
