Param(
  [string]$Name = "ProjectSearchBar"
)

# Build a single-file Windows .exe for the launcher using PyInstaller.
# Usage (PowerShell):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   ./scripts/build_windows_exe.ps1 -Name ProjectSearchBar

$ErrorActionPreference = 'Stop'
Set-Location "$PSScriptRoot"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  Write-Host "Python launcher 'py' not found. Ensure Python 3 is installed and on PATH." -ForegroundColor Yellow
  exit 1
}

try {
  py -3 -m pip install --upgrade pip > $null
  py -3 -m pip install --upgrade pyinstaller > $null
} catch {
  Write-Host "Failed to install PyInstaller: $_" -ForegroundColor Red
  exit 1
}

Write-Host "Building $Name.exe (onefile, windowed)..."
py -3 -m PyInstaller --noconfirm --clean --onefile --windowed --name "$Name" launch.py

Write-Host "Done. Executable at .\dist\$Name.exe"
