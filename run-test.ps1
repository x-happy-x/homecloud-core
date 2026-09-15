$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Run setup.ps1 first' }
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& $python prototype.py scan --photos .\test-photos --models .\models\buffalo_l --data .\data --limit 1000
if ($LASTEXITCODE -ne 0) { throw 'Scan failed' }
$timer.Stop()
Write-Host ('Scan wall time: {0:N1} seconds' -f $timer.Elapsed.TotalSeconds)
& $python prototype.py gallery --data .\data
if ($LASTEXITCODE -ne 0) { throw 'Gallery failed' }
& $python evaluate_sample.py
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
$gallery = Join-Path $PSScriptRoot 'data\gallery.html'
Write-Host "Test passed. Open the gallery: $gallery"
