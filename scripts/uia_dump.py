"""Dump a desktop app's UI Automation tree, for tuning media.ui_play.

Usage (Windows, app open on the screen you want to inspect):
    .venv\\Scripts\\python scripts\\uia_dump.py AppleMusic
Writes data/uia_<process>.txt with one line per control:
    <depth> <ControlType> | name='...' | id='...' | rect=x,y,w,h
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = r"""
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$lines = New-Object System.Collections.Generic.List[string]
function Walk($el, $depth) {
  if ($depth -gt 30 -or $lines.Count -gt 6000) { return }
  $c = $el.Current; $r = $c.BoundingRectangle
  $lines.Add(('{0} {1} | name=''{2}'' | id=''{3}'' | rect={4:N0},{5:N0},{6:N0},{7:N0}' -f
    $depth, $c.ControlType.ProgrammaticName.Replace('ControlType.', ''), $c.Name, $c.AutomationId, $r.X, $r.Y, $r.Width, $r.Height))
  $child = $walker.GetFirstChild($el)
  while ($child) { Walk $child ($depth + 1); $child = $walker.GetNextSibling($child) }
}
foreach ($p in @(Get-Process -Name $env:ASTA_UI_PROCESS -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 })) {
  Walk ($A::FromHandle($p.MainWindowHandle)) 0
}
$lines -join "`n"
"""


def main() -> int:
    process = sys.argv[1] if len(sys.argv) > 1 else "AppleMusic"
    import os

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "ASTA_UI_PROCESS": process},
    )
    text = completed.stdout.strip()
    if not text:
        print(f"No window found for process '{process}'. {completed.stderr.strip()}")
        return 1
    out = Path("data") / f"uia_{process}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text.splitlines())} controls to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
