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
from core.media.catalog import note_play_success
from core.tools.base import Tool

_STOPWORDS = {"the", "a", "an", "of", "by", "song", "songs", "track", "on", "and"}

_FIND_AND_INVOKE = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class AstaDpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
# Report rectangles in real screen pixels, the same space the mouse driver uses
# (a DPI-unaware PowerShell reports scaled coordinates and every click misses).
[AstaDpi]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms
$proc = $env:ASTA_UI_PROCESS
$prefix = $env:ASTA_UI_PREFIX
$tokens = @(($env:ASTA_UI_TOKENS -split '\|') | Where-Object { $_ })
$prefer = @(($env:ASTA_UI_PREFER -split '\|') | Where-Object { $_ })
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
      # The search box's own suggestions echo the typed text verbatim
      # ("baithi hai"); real song rows carry the artist too.
      if ($env:ASTA_UI_SKIP -and $lower.Trim() -eq $env:ASTA_UI_SKIP) { continue }
      foreach ($t in $tokens) { if ($lower.Contains($t)) { $hits++ } }
      if ($hits -lt [Math]::Max(1, [Math]::Ceiling($tokens.Count / 2))) { continue }
      $score = $hits * 1000 - $name.Length
      # A search suggestion is the typed text echoed in lower case
      # ("baithi hai amit trivedi"); a real song row is capitalised.
      if ($name -cmatch '[A-Z]') { $score += 5000 }
      # The artist the catalog resolved to ("Amit Trivedi") beats other versions.
      foreach ($p in $prefer) { if ($lower.Contains($p)) { $score += 400 } }
      try { if ($item.Current.IsOffscreen) { $score -= 3000 } } catch {}
      if ($score -gt $bestScore) { $best = $item; $bestScore = $score; $isItem = $true }
    }
  }
}
function Get-Rect($el) {
  $r = $el.Current.BoundingRectangle
  if ($r.Width -gt 0 -and $r.Height -gt 0 -and -not [double]::IsInfinity($r.X) -and -not [double]::IsInfinity($r.Y)) {
    return @($r.X, $r.Y, $r.Width, $r.Height)
  }
  return $null
}
if ($best -and $isItem) {
  $result.name = [string]$best.Current.Name
  $result.item = $true
  $rect = Get-Rect $best
  if (-not $rect) {
    # Virtualised list: scroll the row into view so it has a real position.
    try { $best.GetCurrentPattern([System.Windows.Automation.ScrollItemPattern]::Pattern).ScrollIntoView(); Start-Sleep -Milliseconds 400 } catch {}
    $rect = Get-Rect $best
  }
  $result.rect = $rect
  # A row's own Play button (shown on hover) beats a double-click.
  if ($env:ASTA_UI_ENTER -ne '1') {
    foreach ($inner in $best.FindAll([System.Windows.Automation.TreeScope]::Descendants, $cond)) {
      $innerName = [string]$inner.Current.Name
      if (-not $innerName -or -not $innerName.ToLower().StartsWith('play')) { continue }
      try {
        $inner.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        $result.invoked = $true; $result.inner = $innerName; break
      } catch {}
    }
  }
  # No usable position (or a second attempt): select the row and press Enter,
  # which plays the selected song and needs no coordinates at all.
  if (-not $result.invoked -and ($env:ASTA_UI_ENTER -eq '1' -or -not $rect)) {
    try { $best.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select() } catch {}
    try { $best.SetFocus(); Start-Sleep -Milliseconds 300; [System.Windows.Forms.SendKeys]::SendWait('{ENTER}'); $result.invoked = $true; $result.inner = 'focus+enter' } catch {}
  }
} elseif ($best) {
  $result.name = [string]$best.Current.Name
  $result.rect = Get-Rect $best
  try {
    $pattern = $best.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke(); $result.invoked = $true
  } catch { $result.error = 'invoke_unsupported' }
} elseif (-not $result.error) { $result.error = 'no_match' }
$result | ConvertTo-Json -Compress
"""

# Type the query into the app's own search box (no deep link available).
# Apple Music collapses its search field into a sidebar icon when the window is
# narrow (and on non-search pages), so there is no Edit control to find until
# search is activated: invoke the Search button, then fall back to Ctrl+F.
_SEARCH_IN_APP = r"""
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class AstaDpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
# Report rectangles in real screen pixels, the same space the mouse driver uses
# (a DPI-unaware PowerShell reports scaled coordinates and every click misses).
[AstaDpi]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms
Add-Type -Namespace AstaWin -Name S -MemberDefinition @'
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
[DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
[DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, UIntPtr e);
'@
$result = @{ found = $false; typed = $false; name = ''; error = ''; activation = ''; candidates = @() }
$windows = @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })
if (-not $windows) { $result.error = 'window_not_found'; $result | ConvertTo-Json -Compress; exit }
$A = [System.Windows.Automation.AutomationElement]
$CT = [System.Windows.Automation.ControlType]
$TS = [System.Windows.Automation.TreeScope]
$editCond = New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::Edit)
$actCond = New-Object System.Windows.Automation.OrCondition(@(
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::Button)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::ListItem)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::TabItem)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::RadioButton)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::Hyperlink)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::MenuItem)),
  (New-Object System.Windows.Automation.PropertyCondition($A::ControlTypeProperty, $CT::Custom))))
$script:fallback = $null

function Focus-App { foreach ($w in $windows) { [AstaWin.S]::SetForegroundWindow($w.MainWindowHandle) | Out-Null } }

function Find-Box {
  foreach ($w in $windows) {
    $edits = $A::FromHandle($w.MainWindowHandle).FindAll($TS::Descendants, $editCond)
    foreach ($e in $edits) {
      $label = ([string]$e.Current.Name + ' ' + [string]$e.Current.AutomationId + ' ' + [string]$e.Current.HelpText).ToLower()
      if ($label.Contains('search')) { return $e }
      if (-not $script:fallback) { $script:fallback = $e }
    }
  }
  return $null
}

# The collapsed search icon: invoke/select it, or click its centre.
function Open-Search {
  foreach ($w in $windows) {
    $root = $A::FromHandle($w.MainWindowHandle)
    foreach ($c in $root.FindAll($TS::Descendants, $actCond)) {
      $label = ([string]$c.Current.Name + ' ' + [string]$c.Current.AutomationId).ToLower()
      if (-not $label.Contains('search')) { continue }
      if ($result.candidates.Count -lt 10) { $result.candidates += ([string]$c.Current.ControlType.ProgrammaticName + ':' + [string]$c.Current.Name) }
      try { $c.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke(); return 'invoke' } catch {}
      try { $c.GetCurrentPattern([System.Windows.Automation.SelectionItemPattern]::Pattern).Select(); return 'select' } catch {}
      $r = $c.Current.BoundingRectangle
      if ($r.Width -gt 0 -and $r.Height -gt 0) {
        [AstaWin.S]::SetCursorPos([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2)) | Out-Null
        Start-Sleep -Milliseconds 80
        [AstaWin.S]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
        [AstaWin.S]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
        return 'click'
      }
    }
  }
  return ''
}

# Whatever just took keyboard focus, when it is a text box of this app.
function Focused-Edit {
  try {
    $f = $A::FocusedElement
    $pids = @($windows | ForEach-Object { $_.Id })
    if ($f -and $f.Current.ControlType.Id -eq $CT::Edit.Id -and $pids -contains $f.Current.ProcessId) { return $f }
  } catch {}
  return $null
}

$box = $null; $how = ''
Focus-App
for ($attempt = 0; $attempt -lt 9 -and -not $box; $attempt++) {
  $box = Find-Box
  if ($box) { break }
  if ($attempt -eq 1) {
    # No text box yet: the search field is collapsed. Open it.
    Focus-App
    $how = Open-Search
    Start-Sleep -Milliseconds 700
    $box = Find-Box
    if (-not $box) { $box = Focused-Edit }
  } elseif ($attempt -eq 4 -or $attempt -eq 7) {
    # Still nothing: Ctrl+F opens and focuses search in Apple Music.
    Focus-App
    [System.Windows.Forms.SendKeys]::SendWait('^f')
    $how = ($how + ' ctrl+f').Trim()
    Start-Sleep -Milliseconds 700
    $box = Find-Box
    if (-not $box) { $box = Focused-Edit }
  }
  if (-not $box) { Start-Sleep -Milliseconds 500 }
}
$result.activation = $how
if (-not $box) { $box = $script:fallback }
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
[DllImport("user32.dll")] public static extern bool IsZoomed(IntPtr h);
[DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
'@
$restored = $false; $maximized = $false
foreach ($w in @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })) {
  if ([AstaWin.U]::IsIconic($w.MainWindowHandle)) { [AstaWin.U]::ShowWindow($w.MainWindowHandle, 9) | Out-Null; $restored = $true }
  # A narrow window collapses the app into its compact layout: the sidebar
  # becomes an overlay that covers the results. Maximise so it stays docked.
  if ($env:ASTA_UI_MAXIMIZE -eq '1' -and -not [AstaWin.U]::IsZoomed($w.MainWindowHandle)) { [AstaWin.U]::ShowWindow($w.MainWindowHandle, 3) | Out-Null; $maximized = $true }
  [AstaWin.U]::SetForegroundWindow($w.MainWindowHandle) | Out-Null
}
if ($restored -or $maximized) { Start-Sleep -Milliseconds 900 }
if ($maximized) { 'maximized' } elseif ($restored) { 'restored' } else { 'ok' }
"""

# Failure diagnostics: what the app exposes to UI Automation.
_DUMP_UI = r"""
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class AstaDpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
# Report rectangles in real screen pixels, the same space the mouse driver uses
# (a DPI-unaware PowerShell reports scaled coordinates and every click misses).
[AstaDpi]::SetProcessDPIAware() | Out-Null
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$w = @(Get-Process -Name $env:ASTA_UI_PROCESS | Where-Object { $_.MainWindowHandle -ne 0 })[0]
if (-not $w) { exit }
$lines = @()
$n = 0
foreach ($e in $A::FromHandle($w.MainWindowHandle).FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)) {
  $name = [string]$e.Current.Name
  if (-not $name) { continue }
  $r = $e.Current.BoundingRectangle
  $lines += ('{0}|{1}|off={2}|{3},{4},{5},{6}' -f $e.Current.ControlType.ProgrammaticName.Replace('ControlType.',''), $name, $e.Current.IsOffscreen, $r.X, $r.Y, $r.Width, $r.Height)
  if (++$n -ge 250) { break }
}
$lines -join "`n"
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


def search_text(query: str) -> str:
    """What to type into an app's search box: "baithi hai by amit" -> "baithi hai amit"."""
    return " ".join(re.sub(r"\b(?:by|bay|bye|buy)\b", " ", core_query(query), flags=re.IGNORECASE).split())


def title_matches(title: str, query: str) -> bool:
    lower = str(title or "").lower()
    tokens = query_tokens(query)
    return bool(tokens) and all(t in lower for t in tokens)


def _ensure_dpi_aware() -> None:
    """Mouse coordinates must be real pixels, matching the rectangles UI Automation reports."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _powershell(script: str, env: dict[str, str], timeout: float) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; " + script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
                    "on_screen": {"type": "boolean"},
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

        session_app = str(args.get("session_app") or "").strip()
        # Apps driven through their search box (Apple Music) need the full layout.
        output["window"] = self._restore_window(process, maximize=bool(session_app and args.get("search")))
        before = self._window_title(process)
        before_session = self._session_title(session_app) if session_app else ""
        searched = bool(args.get("search"))
        # Song rows (not just Play buttons) count when the app shows results
        # we typed, or results the user put on the screen themselves.
        rows = searched or bool(args.get("on_screen"))
        method = None
        if searched and (os.name == "nt" or self.runner is not _powershell):
            typed = self._search_in_app(process, search_text(query))
            output["search"] = typed
            print(f"[UIPlay] {application}: search box={typed.get('name')!r} typed={typed.get('typed')} "
                  f"activation={typed.get('activation') or 'none'} error={typed.get('error') or ''}"
                  + (f" candidates={typed.get('candidates')}" if typed.get("error") else ""), flush=True)
            if not typed.get("typed"):
                return done(False, f"I couldn't find the search box in {application}.")
            self.sleep(2.5)
        prefer = self._preferred_artist_tokens(query)
        if os.name == "nt" or self.runner is not _powershell:
            found = self._invoke_by_name(process, query, items=rows, skip=search_text(query) if searched else "", prefer=prefer)
            core = core_query(query)
            if not found.get("invoked") and not found.get("rect") and core != query:
                # "baithi hai on eppal on apple music": the app name leaked
                # into the title; match the title words alone.
                found = self._invoke_by_name(process, core, attempts=2, items=rows, skip=search_text(query) if searched else "", prefer=prefer)
                output["matched_query"] = core
            if rows and not found.get("invoked") and not found.get("rect"):
                # Results from the network can take a moment to appear.
                self.sleep(2.0)
                found = self._invoke_by_name(process, query, attempts=3, items=rows, skip=search_text(query) if searched else "", prefer=prefer)
                output["retried_lookup"] = True
            output["ui_automation"] = found
            if found.get("invoked"):
                method = "ui_automation"
            elif found.get("rect") and found.get("item"):
                method = self._click_rect(found["rect"], clicks=2) and "ui_automation_double_click"
            elif found.get("rect"):
                method = self._click_rect(found["rect"]) and "ui_automation_click"
        if not method and rows and (os.name == "nt" or self.runner is not _powershell):
            # No usable position for the row: select it and press Enter instead
            # of guessing coordinates with vision.
            via_keys = self._invoke_by_name(
                process, query, attempts=2, items=True, skip=search_text(query) if searched else "",
                enter=True, prefer=prefer,
            )
            if via_keys.get("invoked"):
                output["ui_automation"] = via_keys
                method = "ui_automation_keyboard"
        if not method:
            method = self._vision_fallback(args.get("fallback_target"), application, query)
        output["method"] = method
        print(
            f"[UIPlay] {application}: query={query!r} method={method or 'none'} "
            f"name={(output.get('ui_automation') or {}).get('name')!r}",
            flush=True,
        )
        if not method:
            self._dump_ui(process)
            return done(False, f"Could not find a playable result for '{query}' in {application}.")

        def verify(rounds: int) -> ToolResult | None:
            title = ""
            # Streaming apps (Apple Music) buffer for a few seconds before their
            # media session reports the new track.
            for _ in range(rounds):
                self.sleep(0.7 if session_app else 0.6)
                title = self._window_title(process)
                if title_matches(title, query) or (title and title != before and " - " in title):
                    output["now_playing"] = title
                    output["verified"] = True
                    output["message"] = f"Playing {self._spoken_title(title)}."
                    note_play_success(query)
                    return done(True)
                if session_app:
                    # Apps whose window title never changes (Apple Music):
                    # confirm through the Windows media session instead.
                    info = self._session(session_app)
                    now = str(info.get("title") or "")
                    status = str(info.get("status") or "").lower()
                    playing = status == "playing"
                    named = title_matches(f"{now} {info.get('artist') or ''}", core_query(query))
                    output["session"] = {"title": now, "status": status, "app": info.get("app")}
                    if now and ((named and status not in {"stopped", "closed", ""}) or (playing and now != before_session)):
                        artist = str(info.get("artist") or "").strip()
                        output["now_playing"] = f"{artist} - {now}" if artist else now
                        output["verified"] = True
                        output["message"] = f"Playing {now}" + (f" by {artist}." if artist else ".")
                        note_play_success(query, now, artist)
                        return done(True)
            output["now_playing"] = title
            return None

        result = verify(12 if session_app else 6)
        if result:
            return result
        if rows and method not in {"ui_automation", "ui_automation_keyboard"} and (os.name == "nt" or self.runner is not _powershell):
            # Second, coordinate-free attempt: select the result row and press Enter.
            again = self._invoke_by_name(
                process, query, attempts=2, items=True, skip=search_text(query) if searched else "",
                enter=True, prefer=prefer,
            )
            output["keyboard_retry"] = again
            print(f"[UIPlay] {application}: keyboard retry name={again.get('name')!r} invoked={again.get('invoked')}", flush=True)
            if again.get("invoked"):
                result = verify(10 if session_app else 5)
                if result:
                    return result
        title = str(output.get("now_playing") or "")
        output["now_playing"] = title
        output["verified"] = False
        self._dump_ui(process)
        if session_app:
            print(f"[UIPlay] {application}: media session after play: {output.get('session')}", flush=True)
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
            raw = self.runner(_SEARCH_IN_APP, env, 30.0)
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

    def _preferred_artist_tokens(self, query: str) -> tuple[str, ...]:
        """Artist words from the catalog match, so the right version of a song wins."""
        try:
            from core.media.catalog import get_resolver, primary_artist

            last = get_resolver().last
            if last and last.query.strip().lower() == str(query).strip().lower() and last.artist:
                return tuple(w for w in query_tokens(primary_artist(last.artist)) if len(w) > 2)
        except Exception:
            pass
        return ()

    def _dump_ui(self, process: str) -> None:
        """On failure, save what the app's accessibility tree exposes, for diagnosis."""
        if not (os.name == "nt" or self.runner is not _powershell):
            return
        try:
            raw = self.runner(_DUMP_UI, {"ASTA_UI_PROCESS": process}, 15.0)
            if not str(raw or "").strip():
                return
            from pathlib import Path

            path = Path(__file__).resolve().parents[2] / "data" / f"ui_dump_{process}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(raw), encoding="utf-8")
            print(f"[UIPlay] Saved the {process} UI tree to {path}", flush=True)
        except Exception:
            pass

    def _invoke_by_name(
        self, process: str, query: str, attempts: int = 6, items: bool = False, skip: str = "",
        enter: bool = False, prefer: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        env = {
            "ASTA_UI_ENTER": "1" if enter else "0",
            "ASTA_UI_PREFER": "|".join(prefer),
            "ASTA_UI_ITEMS": "1" if items else "0",
            "ASTA_UI_ATTEMPTS": str(attempts),
            "ASTA_UI_PROCESS": process,
            "ASTA_UI_PREFIX": "Play",
            "ASTA_UI_TOKENS": "|".join(query_tokens(query)),
            "ASTA_UI_SKIP": " ".join(str(skip or "").lower().split()),
        }
        try:
            raw = self.runner(_FIND_AND_INVOKE, env, 20.0)
            data = json.loads(raw.splitlines()[-1]) if raw else {}
        except Exception as exc:
            return {"invoked": False, "error": f"{type(exc).__name__}: {exc}"}
        return data if isinstance(data, dict) else {}

    def _restore_window(self, process: str, maximize: bool = False) -> str:
        if not (os.name == "nt" or self.runner is not _powershell):
            return "skipped"
        if os.getenv("ASTA_UI_MAXIMIZE", "1").strip().lower() in {"0", "false", "no", "off"}:
            maximize = False
        try:
            raw = self.runner(
                _RESTORE_WINDOW,
                {"ASTA_UI_PROCESS": process, "ASTA_UI_MAXIMIZE": "1" if maximize else "0"},
                10.0,
            )
        except Exception:
            return "unknown"
        state = (str(raw or "").strip().splitlines() or ["unknown"])[-1]
        if state == "restored":
            print(f"[UIPlay] Restored minimized {process} window.", flush=True)
        elif state == "maximized":
            print(f"[UIPlay] Maximised the {process} window so its sidebar stays docked.", flush=True)
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
        _ensure_dpi_aware()
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
