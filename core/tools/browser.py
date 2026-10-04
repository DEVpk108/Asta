"""Open a web search directly in a browser.

Typing into a browser window depends on focus and on whatever the window
shows first: Chrome's "Who's using Chrome?" profile picker swallows Ctrl+L
and the typed query. Launching the browser with the search URL (and, for
Chromium browsers, an explicit profile) skips the picker, needs no vision
model, and works whether or not the browser is already running.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote_plus

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool

DEFAULT_SEARCH_URL = "https://www.google.com/search?q={query}"

# name -> (executables, Windows install paths relative to env roots, user-data dir)
_BROWSERS = {
    "chrome": (
        ("chrome", "chrome.exe", "google-chrome", "google-chrome-stable"),
        ("Google/Chrome/Application/chrome.exe",),
        ("LOCALAPPDATA", "Google/Chrome/User Data"),
    ),
    "edge": (
        ("msedge", "msedge.exe", "microsoft-edge"),
        ("Microsoft/Edge/Application/msedge.exe",),
        ("LOCALAPPDATA", "Microsoft/Edge/User Data"),
    ),
    "brave": (
        ("brave", "brave.exe", "brave-browser"),
        ("BraveSoftware/Brave-Browser/Application/brave.exe",),
        ("LOCALAPPDATA", "BraveSoftware/Brave-Browser/User Data"),
    ),
    "firefox": (
        ("firefox", "firefox.exe"),
        ("Mozilla Firefox/firefox.exe",),
        None,
    ),
}

_ALIASES = {
    "chrome": "chrome", "google chrome": "chrome", "chromium": "chrome",
    "edge": "edge", "microsoft edge": "edge", "msedge": "edge",
    "brave": "brave", "brave browser": "brave",
    "firefox": "firefox", "mozilla firefox": "firefox",
}


def browser_key(name: str) -> str | None:
    value = " ".join(str(name or "").lower().replace(".exe", "").split())
    return _ALIASES.get(value)


def search_url(query: str) -> str:
    template = os.getenv("ASTA_SEARCH_URL", DEFAULT_SEARCH_URL)
    if "{query}" not in template:
        template = DEFAULT_SEARCH_URL
    return template.replace("{query}", quote_plus(str(query).strip()))


def _find_executable(key: str) -> str | None:
    names, relative_paths, _ = _BROWSERS[key]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for root_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(root_var)
        if not root:
            continue
        for relative in relative_paths:
            candidate = Path(root) / relative
            if candidate.exists():
                return str(candidate)
    return None


def chromium_profile(key: str) -> str | None:
    """Profile to open: ASTA_BROWSER_PROFILE, else the browser's last used."""
    configured = os.getenv("ASTA_BROWSER_PROFILE", "").strip()
    if configured:
        return configured
    data = _BROWSERS.get(key, (None, None, None))[2]
    if not data:
        return None
    root = os.environ.get(data[0])
    if not root:
        return None
    local_state = Path(root) / data[1] / "Local State"
    try:
        state = json.loads(local_state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Default" if (Path(root) / data[1] / "Default").exists() else None
    profile = state.get("profile") or {}
    last = profile.get("last_used")
    if isinstance(last, str) and last.strip():
        return last.strip()
    return "Default"


class BrowserSearchTool(Tool):
    """Open a search results page in the requested (or default) browser."""

    def __init__(self, popen=None, startfile=None):
        self._popen = popen or subprocess.Popen
        self._startfile = startfile or getattr(os, "startfile", None)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser.search",
            description="Open web search results for a query in a browser.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "browser": {"type": "string"},
                },
                "required": ["query"],
            },
            risk_level="low",
            requires_confirmation=False,
            timeout_seconds=10.0,
            metadata={"actions": ["web_search"], "category": "browser"},
        )

    def build_command(self, query: str, browser: str | None) -> tuple[list[str] | None, str]:
        url = search_url(query)
        key = browser_key(browser or "")
        if key is None:
            return None, url
        executable = _find_executable(key)
        if executable is None:
            return None, url
        command = [executable]
        if key != "firefox":
            profile = chromium_profile(key)
            if profile:
                command.append(f"--profile-directory={profile}")
        else:
            command.append("--new-tab")
        command.append(url)
        return command, url

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        query = str(request.arguments.get("query") or "").strip()
        browser = str(request.arguments.get("browser") or "").strip() or None
        if not query:
            return ToolResult(success=False, tool=self.definition.name, error="query must be non-empty.")
        try:
            command, url = self.build_command(query, browser)
            if command:
                self._popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                method = "browser"
            elif self._startfile is not None:
                self._startfile(url)
                method = "default_browser"
            else:
                import webbrowser

                webbrowser.open(url)
                method = "webbrowser"
        except Exception as exc:
            return ToolResult(
                success=False,
                tool=self.definition.name,
                error=f"Could not open the search: {type(exc).__name__}: {exc}",
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )
        print(f"[Browser] Search {query!r} via {method}: {command or url}", flush=True)
        return ToolResult(
            success=True,
            tool=self.definition.name,
            output={"query": query, "browser": browser, "url": url, "method": method},
            duration_seconds=time.perf_counter() - start,
            metadata={"request_id": request.request_id},
        )
