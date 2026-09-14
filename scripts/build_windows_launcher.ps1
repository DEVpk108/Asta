$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $python) {
    throw 'Python was not found. Create/activate the A.S.T.A. virtual environment first.'
}

& $python -m pip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller installation failed.' }

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name ASTA `
    scripts\asta_launcher.py

if ($LASTEXITCODE -ne 0) { throw 'A.S.T.A. launcher build failed.' }

Write-Host ''
Write-Host 'A.S.T.A. launcher built successfully:'
Write-Host (Join-Path $root 'dist\ASTA.exe')
Write-Host ''
Write-Host 'Place ASTA.exe in the A.S.T.A. project root next to main.py and .venv before the demo.'
