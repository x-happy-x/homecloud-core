<#
  Бэкенд локальной фототеки: только API и медиа.
  Веб-интерфейс живёт на Proxmox (/opt/homecloud) и ходит сюда по сети.
#>
param(
    [string]$Catalog = (Join-Path $PSScriptRoot 'trash-clean-catalog'),
    [int]$Port = 18311,
    [string]$ListenAddress = '0.0.0.0',
    [int]$MinClusterSize = 8,
    [int]$MaxFaces = 100000,   # каталог давно больше прототипного образца
    [string]$TokenFile = (Join-Path $PSScriptRoot 'backend-token.txt'),
    [string]$DeviceId = ($env:COMPUTERNAME.ToLowerInvariant()),
    [string]$DeviceName = $env:COMPUTERNAME
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Run setup.ps1 first' }
if (!(Test-Path -LiteralPath (Join-Path $Catalog 'catalog.sqlite'))) {
    throw "Catalog not found: $Catalog"
}
# Без GPU-провайдера этапы с моделями не сработают, а узнать об этом
# посреди сканирования — дорого.
try {
    $providers = & $python -c "import onnxruntime as o; print(','.join(o.get_available_providers()))"
} catch { $providers = 'проверить не удалось' }
if ($providers -notmatch 'CUDAExecutionProvider') {
    Write-Warning "onnxruntime не видит видеокарту (провайдеры: $providers). Лица не посчитаются — запустите setup.ps1."
}

& $python web_server.py --data $Catalog --host $ListenAddress --port $Port `
    --min-cluster-size $MinClusterSize --max-faces $MaxFaces `
    --token-file $TokenFile --no-browser `
    --device-id $DeviceId --device-name $DeviceName
