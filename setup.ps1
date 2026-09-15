$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$envPath = Join-Path $workspace 'work\v'
$env:PIP_CACHE_DIR = Join-Path $workspace 'work\pip-cache'
New-Item -ItemType Directory -Force (Split-Path -Parent $envPath), $env:PIP_CACHE_DIR | Out-Null
python -m venv $envPath
if ($LASTEXITCODE -ne 0) { throw 'Cannot create virtual environment' }
$python = Join-Path $envPath 'Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Cannot install pip' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependencies not installed. See README.md for Windows build requirements.' }
& $python -m pip uninstall -y onnxruntime
& $python -m pip install --force-reinstall --no-deps 'onnxruntime-gpu>=1.21,<1.27'
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime installation failed' }
& $python prototype.py doctor
Write-Host "Ready. Python environment: $envPath"
