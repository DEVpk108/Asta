"""Play a search result through the app's accessibility tree.

Vision grounding with a small VL model is unreliable for dense result lists,
so for apps that expose Windows UI Automation (Spotify's desktop app does),
press the result's own "Play <title>" button by name and confirm playback
from the window title (Spotify shows "Artist - Title" while playing). Vision
locate + double-click remains the fallback.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any, Callable

from core.contracts import ToolDefinition, ToolRequest, ToolResult
from core.tools.base import Tool

_STOPWORDS = {"the", "a", "an", "of", "by", "song", "songs", "track", "on", "and"}

_FIND_AND_INVOKE = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$proc = $env:ASTA_UI_PROCESS
$prefix = $env:ASTA_UI_PREFIX
$tokens = @(($env:ASTA_UI_TOKENS -split '\|') | Where-Object { $_ })
$result = @{ invoked = $false; name = ''; candidates = @(); rect = $null; error = '' }
$windows = @(Get-Process -Name $proc -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })
if (-not $windows) { $result.error = 'window_not_found'; $result | ConvertTo-Json -Compress; exit }
$A = [System.Windows.Automation.AutomationElement]
$cond = New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
$best = $null; $bestScore = -1
$maxAttempts = [int]($env:ASTA_UI_ATTEMPTS); if ($maxAttempts -lt 1) { $maxAttempts = 6 }
for ($attempt = 0; $attempt -lt $maxAttempts -and -not $best; $attempt++) {
  foreach ($w in $windows) {
    $root = $A::FromHandle($w.MainWindowHandle)
    $buttons = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    foreach ($b in $buttons) {
      $name = [string]$b.Current.Name
      if (-not $name -or -not $name.ToLower().StartsWith($prefix.ToLower())) { continue }
      if ($result.candidates.Count -lt 12) { $result.candidates += $name }
      $lower = $name.ToLower(); $hits = 0
      foreach ($t in $tokens) { if ($lower.Contains($t)) { $hits++ } }
      # All words match best; otherwise accept the result matching most words
      # (at least half), earliest in the list on ties.
      if ($hits -lt [Math]::Max(1, [Math]::Ceiling($tokens.Count / 2))) { continue }
      $score = $hits * 1000 - $name.Length
      if ($hits -lt $tokens.Count) { $score = $hits * 1000 - 500 }
      if ($score -gt $bestScore) { $best = $b; $bestScore = $score }
    }
  }
  # Chromium/CEF builds its accessibility tree on the first UIA query.
  if (-not $best) { Start-Sleep -Milliseconds 800 }
}
$isItem = $false
if (-not $best -and $env:ASTA_UI_ITEMS -eq '1') {
  # Apps without per-result Play buttons (Apple Music): a song row whose
  # name has the query words; the caller double-clicks it.
  $CT = [System.Windows.Automation.ControlType]
  $itemCond = New-Object System.Windows.Automation.OrCondition(@(
    (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::ListItem)),
    (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::DataItem)),
    (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::TreeItem))))
  foreach ($w in $windows) {
    $root = $A::FromHandle($w.MainWindowHandle)
    foreach ($item in $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $itemCond)) {
      $name = [string]$item.Current.Name
      if (-not $name) { continue }
      $lower = $name.ToLower(); $hits = 0
      foreach ($t in $tokens) { if ($lower.Contains($t)) { $hits++ } }
      if ($hits -lt [Math]::Max(1, [Math]::Ceiling($tokens.Count / 2))) { continue }
      $score = $hits * 1000 - $name.Length
      if ($score -gt $bestScore) { $best = $item; $bestScore = $score; $isItem = $true }
    }
  }
}
if ($best -and $isItem) {
  $result.name = [string]$best.Current.Name
  $r = $best.Current.BoundingRectangle
  $result.rect = @($r.X, $r.Y, $r.Width, $r.Height)
  $result.item = $true
} elseif ($best) {
  $result.name = [string]$best.Current.Name
  $r = $best.Current.BoundingRectangle
  $result.rect = @($r.X, $r.Y, $r.Width, $r.Height)
  try {
    $pattern = $best.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke(); $result.invoked = $true
  } catch { $result.error = 'invoke_unsupported' }
} elseif (-not $result.error) { $result.error = 'no_match' }
$result | ConvertTo-Json -Compress
"""

# Type the query into the app's own search box (no deep link available).
_SEARCH_IN_APP = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms
$result = @{ found = $false; typed = $false; name = ''; error = '' }
$windows = @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })
if (-not $windows) { $result.error = 'window_not_found'; $result | ConvertTo-Json -Compress; exit }
$A = [System.Windows.Automation.AutomationElement]
$cond = New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, [System.Windows.Automation.ControlType]::Edit)
$box = $null; $fallback = $null
for ($attempt = 0; $attempt -lt 8 -and -not $box; $attempt++) {
  foreach ($w in $windows) {
    $edits = $A::FromHandle($w.MainWindowHandle).FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)
    foreach ($e in $edits) {
      $label = ([string]$e.Current.Name + ' ' + [string]$e.Current.AutomationId + ' ' + [string]$e.Current.HelpText).ToLower()
      if ($label.Contains('search')) { $box = $e; break }
      if (-not $fallback) { $fallback = $e }
    }
    if ($box) { break }
  }
  if (-not $box) { Start-Sleep -Milliseconds 700 }
}
if (-not $box) { $box = $fallback }
if (-not $box) { $result.error = 'search_box_not_found'; $result | ConvertTo-Json -Compress; exit }
$result.found = $true; $result.name = [string]$box.Current.Name
try { $box.SetFocus() } catch {}
Start-Sleep -Milliseconds 250
try {
  $vp = $box.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
  $vp.SetValue($env:ASTA_UI_QUERY); $result.typed = $true
} catch {}
if (-not $result.typed) {
  [System.Windows.Forms.SendKeys]::SendWait('^a')
  [System.Windows.Forms.SendKeys]::SendWait($env:ASTA_UI_KEYS); $result.typed = $true
}
Start-Sleep -Milliseconds 400
try { $box.SetFocus() } catch {}
[System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
$result | ConvertTo-Json -Compress
"""


def sendkeys_escape(text: str) -> str:
    """Literal text for SendKeys (+ ^ % ~ ( ) { } [ ] are commands)."""
    return re.sub(r"([+^%~(){}\[\]])", r"{\1}", str(text or ""))


# A minimized window has no usable accessibility tree or pixels, so bring it
# back (SW_RESTORE) and to the front before looking for the Play button.
_RESTORE_WINDOW = r"""
Add-Type -Namespace AstaWin -Name U -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
[DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
'@
$restored = $false
foreach ($w in @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })) {
  if ([AstaWin.U]::IsIconic($w.MainWindowHandle)) { [AstaWin.U]::ShowWindow($w.MainWindowHandle, 9) | Out-Null; $restored = $true }
  [AstaWin.U]::SetForegroundWindow($w.MainWindowHandle) | Out-Null
}
if ($restored) { Start-Sleep -Milliseconds 700; 'restored' } else { 'ok' }
"""

_WINDOW_TITLE = r"""
$t = @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue |
  Where-Object { $_.MainWindowTitle } | Select-Object -ExpandProperty MainWindowTitle)
($t -join "`n")
"""


def query_tokens(query: str) -> list[str]:
    words = re.findall(r"[\w']+", str(query or "").lower())
    tokens = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return tokens or words


def core_query(query: str) -> str:
    """Title words before a leaked "on <app>" tail."""
    value = " ".join(str(query or "").split())
    head = re.split(r"\s+(?:on|in|using)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return head if len(head) >= 2 else value


def title_matches(title: str, query: str) -> bool:
    lower = str(title or "").lower()
    tokens = query_tokens(query)
    return bool(tokens) and all(t in lower for t in tokens)


def _powershell(script: str, env: dict[str, str], timeout: float) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **env},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    return (completed.stdout or "").strip()


class UIPlayTool(Tool):
    """Start a search result playing via UI Automation, vision as fallback."""

    def __init__(
        self,
        *,
        runner: Callable[[str, dict[str, str], float], str] | None = None,
        locate_tool=None,
        controller=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.runner = runner or _powershell
        self.locate_tool = locate_tool
        self.controller = controller
        self.sleep = sleep

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="media.ui_play",
            description=(
                "Play a visible search result in a desktop media app by pressing "
                "its accessible Play button, falling back to vision, and confirm "
                "playback from the window title."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "application": {"type": "string"},
                    "process": {"type": "string"},
                    "fallback_target": {"type": "string"},
                    "search": {"type": "boolean"},
                    "session_app": {"type": "string"},
                },
                "required": ["query"],
            },
            risk_level="medium",
            timeout_seconds=40.0,
            metadata={"actions": ["ui_play"], "category": "media"},
        )

    def execute(self, request: ToolRequest) -> ToolResult:
        start = time.perf_counter()
        args = request.arguments or {}
        query = str(args.get("query") or "").strip()
        application = str(args.get("application") or "Spotify").strip()
        process = str(args.get("process") or application).strip()
        output: dict[str, Any] = {"query": query, "application": application}

        def done(success: bool, error: str | None = None) -> ToolResult:
            return ToolResult(
                success=success,
                tool=request.tool,
                output=output,
                error=error,
                duration_seconds=time.perf_counter() - start,
                metadata={"request_id": request.request_id},
            )

        if not query:
            return done(False, "Argument 'query' must be a non-empty string.")

        output["window"] = self._restore_window(process)
        before = self._window_title(process)
        session_app = str(args.get("session_app") or "").strip()
        before_session = self._session_title(session_app) if session_app else ""
        searched = bool(args.get("search"))
        method = None
        if searched and (os.name == "nt" or self.runner is not _powershell):
            typed = self._search_in_app(process, query)
            output["search"] = typed
            print(f"[UIPlay] {application}: search box={typed.get('name')!r} typed={typed.get('typed')} "
                  f"error={typed.get('error') or ''}", flush=True)
            if not typed.get("typed"):
                return done(False, f"I couldn't find the search box in {application}.")
            self.sleep(2.5)
        if os.name == "nt" or self.runner is not _powershell:
            found = self._invoke_by_name(process, query, items=searched)
            core = core_query(query)
            if not found.get("invoked") and not found.get("rect") and core != query:
                # "baithi hai on eppal on apple music": the app name leaked
                # into the title; match the title words alone.
                found = self._invoke_by_name(process, core, attempts=2, items=searched)
                output["matched_query"] = core
            output["ui_automation"] = found
            if found.get("invoked"):
                method = "ui_automation"
            elif found.get("rect") and found.get("item"):
                method = self._click_rect(found["rect"], clicks=2) and "ui_automation_double_click"
            elif found.get("rect"):
                method = self._click_rect(found["rect"]) and "ui_automation_click"
        if not method:
            method = self._vision_fallback(args.get("fallback_target"), application, query)
        output["method"] = method
        print(
            f"[UIPlay] {application}: query={query!r} method={method or 'none'} "
            f"name={(output.get('ui_automation') or {}).get('name')!r}",
            flush=True,
        )
        if not method:
            return done(False, f"Could not find a playable result for '{query}' in {application}.")

        title = ""
        for _ in range(6):
            self.sleep(0.6)
            title = self._window_title(process)
            if title_matches(title, query) or (title and title != before and " - " in title):
                output["now_playing"] = title
                output["verified"] = True
                output["message"] = f"Playing {self._spoken_title(title)}."
                return done(True)
            if session_app:
                # Apps whose window title never changes (Apple Music):
                # confirm through the Windows media session instead.
                info = self._session(session_app)
                now = str(info.get("title") or "")
                playing = str(info.get("status") or "").lower() == "playing"
                if now and playing and (title_matches(f"{now} {info.get('artist') or ''}", query) or now != before_session):
                    artist = str(info.get("artist") or "").strip()
                    output["now_playing"] = f"{artist} - {now}" if artist else now
                    output["verified"] = True
                    output["message"] = f"Playing {now}" + (f" by {artist}." if artist else ".")
                    return done(True)
        output["now_playing"] = title
        output["verified"] = False
        return done(
            False,
            f"Pressed play for '{query}' in {application}, but playback did not start "
            f"(window title: {title or 'unknown'}).",
        )

    # -- helpers -------------------------------------------------------

    def _search_in_app(self, process: str, query: str) -> dict[str, Any]:
        env = {
            "ASTA_UI_PROCESS": process,
            "ASTA_UI_QUERY": query,
            "ASTA_UI_KEYS": sendkeys_escape(query),
        }
        try:
            raw = self.runner(_SEARCH_IN_APP, env, 20.0)
            data = json.loads(raw.splitlines()[-1]) if raw else {}
        except Exception as exc:
            return {"typed": False, "error": f"{type(exc).__name__}: {exc}"}
        return data if isinstance(data, dict) else {}

    def _session(self, app: str) -> dict[str, Any]:
        from core.media.smtc import media_session

        runner = None if self.runner is _powershell else self.runner
        try:
            return media_session("", app, runner=runner) or {}
        except Exception:
            return {}

    def _session_title(self, app: str) -> str:
        return str(self._session(app).get("title") or "")

    def _invoke_by_name(self, process: str, query: str, attempts: int = 6, items: bool = False) -> dict[str, Any]:
        env = {
            "ASTA_UI_ITEMS": "1" if items else "0",
            "ASTA_UI_ATTEMPTS": str(attempts),
            "ASTA_UI_PROCESS": process,
            "ASTA_UI_PREFIX": "Play",
            "ASTA_UI_TOKENS": "|".join(query_tokens(query)),
        }
        try:
            raw = self.runner(_FIND_AND_INVOKE, env, 20.0)
            data = json.loads(raw.splitlines()[-1]) if raw else {}
        except Exception as exc:
            return {"invoked": False, "error": f"{type(exc).__name__}: {exc}"}
        return data if isinstance(data, dict) else {}

    def _restore_window(self, process: str) -> str:
        if not (os.name == "nt" or self.runner is not _powershell):
            return "skipped"
        try:
            raw = self.runner(_RESTORE_WINDOW, {"ASTA_UI_PROCESS": process}, 8.0)
        except Exception:
            return "unknown"
        state = (str(raw or "").strip().splitlines() or ["unknown"])[-1]
        if state == "restored":
            print(f"[UIPlay] Restored minimized {process} window.", flush=True)
        return state

    def _window_title(self, process: str) -> str:
        try:
            raw = self.runner(_WINDOW_TITLE, {"ASTA_UI_PROCESS": process}, 8.0)
        except Exception:
            return ""
        lines = [line.strip() for line in str(raw or "").splitlines() if line.strip()]
        return lines[0] if lines else ""

    def _click_rect(self, rect, clicks: int = 1) -> bool:
        if self.controller is None:
            return False
        try:
            x, y, w, h = (float(v) for v in rect)
            if w <= 0 or h <= 0:
                return False
            if clicks > 1:
                self.controller.click(x=x + w / 2, y=y + h / 2, clicks=clicks, interval=0.08)
            else:
                self.controller.click(x=x + w / 2, y=y + h / 2)
            return True
        except Exception:
            return False

    def _vision_fallback(self, target, application: str, query: str):
        if self.locate_tool is None or self.controller is None:
            return None
        target = str(target or f"the first song in the search results in {application}")
        try:
            result = self.locate_tool.execute(
                ToolRequest(
                    tool="vision.locate",
                    arguments={"target": target},
                    request_id="ui-play-fallback",
                )
            )
        except Exception:
            return None
        output = (result.output or {}) if result.success else {}
        center = output.get("screen_center") or output.get("center") or {}
        x, y = center.get("x"), center.get("y")
        if x is None or y is None:
            return None
        try:
            self.controller.click(x=x, y=y, clicks=2, interval=0.08)
        except Exception:
            return None
        return "vision_double_click"

    @staticmethod
    def _spoken_title(title: str) -> str:
        # Spotify: "Artist - Title" -> "Title by Artist".
        if " - " in title:
            artist, _, song = title.partition(" - ")
            return f"{song.strip()} by {artist.strip()}"
        return title
