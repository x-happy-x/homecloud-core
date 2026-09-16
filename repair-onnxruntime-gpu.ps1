<#
  Repairs Python environments where a dependency pulled the CPU-only
  onnxruntime package over onnxruntime-gpu.
#>
param(
    [string[]]$EnvPaths = @(
        (Join-Path $PSScriptRoot '.venv'),
        (Join-Path $PSScriptRoot 'work\v'),
        (Join-Path $PSScriptRoot 'work\vision-venv')
    )
)

$ErrorActionPreference = 'Stop'

foreach ($envPath in $EnvPaths) {
    $python = Join-Path $envPath 'Scripts\python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        Write-Warning "Skip missing environment: $envPath"
        continue
    }

    Write-Host "-> $envPath"
    & $python -m pip uninstall -y onnxruntime
    & $python -m pip install --force-reinstall --no-deps 'onnxruntime-gpu>=1.21,<1.27'
    if ($LASTEXITCODE -ne 0) { throw "Cannot install onnxruntime-gpu in $envPath" }

    $providers = & $python -c "import onnxruntime as o; print(','.join(o.get_available_providers()))"
    if ($providers -notmatch 'CUDAExecutionProvider') {
        throw "CUDAExecutionProvider is missing in $envPath; providers: $providers"
    }
    Write-Host "   providers: $providers"
}
