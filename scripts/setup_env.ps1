<#
PowerShell helper to create a virtual environment and install project requirements.

Usage (from repo root):
  .\scripts\setup_env.ps1

This will create `/.venv`, upgrade packaging tools, and install packages from
`requirements.txt` into the venv Python. Run in an elevated shell if you hit
permission issues.
#>

$ErrorActionPreference = 'Stop'

$venvPath = Join-Path -Path (Get-Location) -ChildPath '.venv'
$python = Join-Path -Path $venvPath -ChildPath 'Scripts\python.exe'

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

Write-Host "Upgrading pip, setuptools, wheel in venv..."
& $python -m pip install --upgrade pip setuptools wheel

if (-Not (Test-Path 'requirements.txt')) {
    Write-Error "requirements.txt not found in repo root. Aborting."
    exit 1
}

Write-Host "Installing project requirements into venv (this may take several minutes)..."
& $python -m pip install --require-hashes -r requirements.txt

Write-Host "Setup complete. To activate the venv in PowerShell run:`n  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: ollama pull qwen3:1.7b; ollama pull nomic-embed-text; python -m vet_ai_tutor.api"