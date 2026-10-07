"""Windows media-session control (no vision, no app API).

Uses the Windows System Media Transport Controls (the same session the
volume flyout shows) through Windows PowerShell's WinRT projection, so pause
really pauses, next/previous really skip, and the now-playing title/artist
come back for a spoken confirmation. Works for Spotify, browsers, Apple Music
and any app that publishes a media session.
"""
from __future__ import annotations

import json
import os
import subprocess

_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
  $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$type) {
  $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
  $null = $task.Wait(5000); $task.Result
}
$null = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType=WindowsRuntime]
$M = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]
$P = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties]
$out = @{ ok = $false; app = ''; title = ''; artist = ''; status = ''; error = '' }
try { $mgr = Await ($M::RequestAsync()) $M } catch { $out.error = 'smtc_unavailable'; $out | ConvertTo-Json -Compress; exit }
$hint = [string]$env:ASTA_MEDIA_APP
$session = $null
if ($hint) {
  foreach ($s in $mgr.GetSessions()) {
    if ([string]$s.SourceAppUserModelId -and ([string]$s.SourceAppUserModelId).ToLower().Contains($hint.ToLower())) { $session = $s; break }
  }
}
if (-not $session) { $session = $mgr.GetCurrentSession() }
if (-not $session) { $out.error = 'no_session'; $out | ConvertTo-Json -Compress; exit }
$out.app = [string]$session.SourceAppUserModelId
function Title() { try { (Await ($session.TryGetMediaPropertiesAsync()) $P).Title } catch { '' } }
$op = [string]$env:ASTA_MEDIA_OP
$before = Title
$ok = $true
switch ($op) {
  'play'     { $ok = Await ($session.TryPlayAsync()) ([bool]) }
  'pause'    { $ok = Await ($session.TryPauseAsync()) ([bool]) }
  'toggle'   { $ok = Await ($session.TryTogglePlayPauseAsync()) ([bool]) }
  'next'     { $ok = Await ($session.TrySkipNextAsync()) ([bool]) }
  'previous' { $ok = Await ($session.TrySkipPreviousAsync()) ([bool]) }
  'stop'     { $ok = Await ($session.TryStopAsync()) ([bool]) }
}
if ($op -in @('next', 'previous')) {
  Start-Sleep -Milliseconds 900
  # Players restart the current track on the first "previous" once it has
  # played a few seconds; press again to really go back.
  if ($op -eq 'previous' -and (Title) -eq $before) {
    $ok = Await ($session.TrySkipPreviousAsync()) ([bool]); Start-Sleep -Milliseconds 900
  }
} elseif ($op) { Start-Sleep -Milliseconds 350 }
try { $props = Await ($session.TryGetMediaPropertiesAsync()) $P; $out.title = [string]$props.Title; $out.artist = [string]$props.Artist } catch {}
try { $out.status = [string]$session.GetPlaybackInfo().PlaybackStatus } catch {}
$out.ok = [bool]$ok
$out | ConvertTo-Json -Compress
"""


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


def media_session(operation: str = "", app: str = "", *, runner=None) -> dict:
    """Run one transport operation ('' = just read) and return session info."""
    run = runner or _powershell
    try:
        raw = run(
            _SCRIPT,
            {"ASTA_MEDIA_OP": str(operation or ""), "ASTA_MEDIA_APP": str(app or "")},
            15.0,
        )
        data = json.loads(raw.splitlines()[-1]) if raw else {}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"ok": False, "error": "invalid_output"}


def describe(info: dict) -> str:
    title = str(info.get("title") or "").strip()
    artist = str(info.get("artist") or "").strip()
    if not title:
        return ""
    return f"{title} by {artist}" if artist else title


def spoken_message(operation: str, info: dict) -> str:
    track = describe(info)
    status = str(info.get("status") or "").lower()
    if operation == "pause":
        return f"Paused {track}." if track else "Paused."
    if operation == "stop":
        return "Stopped playback."
    if operation in {"play", "toggle"}:
        if status == "paused":
            return f"Paused {track}." if track else "Paused."
        return f"Resumed {track}." if track else "Resumed playback."
    if operation == "next":
        return f"Now playing {track}." if track else "Skipped to the next track."
    if operation == "previous":
        return f"Back to {track}." if track else "Went back to the previous track."
    if operation == "now_playing":
        if not track:
            return "Nothing is playing right now."
        if status == "paused":
            return f"{track} is paused."
        return f"This is {track}."
    return "Done."
