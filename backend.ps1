<#
  Бэкенд HomeCloud на этом компьютере.

  Ядро при хабе (есть core.json): только вычисления — каталог, превью и лица
  живут на сервере HomeCloud, адрес и токен связи лежат в core-data\hub.json.
  Их пишет Install-Core.ps1, когда ядро ставят или обновляют из интерфейса.

  Без core.json — прежний бэкенд одного компьютера со своим каталогом.
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
# Без GPU-провайдера этапы с моделями не сработают, а узнать об этом
# посреди сканирования — дорого.
try {
    $providers = & $python -c "import onnxruntime as o; print(','.join(o.get_available_providers()))"
} catch { $providers = 'проверить не удалось' }
if ($providers -notmatch 'CUDAExecutionProvider') {
    Write-Warning "onnxruntime не видит видеокарту (провайдеры: $providers). Лица не посчитаются — запустите setup.ps1."
}

$coreConfig = Join-Path $PSScriptRoot 'core.json'
if (Test-Path -LiteralPath $coreConfig) {
    $core = Get-Content -LiteralPath $coreConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $data = Join-Path $PSScriptRoot 'core-data'
    $arguments = @('web_server.py', '--role', 'core', '--data', $data,
        '--host', $ListenAddress, '--port', $Port, '--token-file', $TokenFile,
        '--device-id', $core.id, '--device-name', $core.name)
    # Старый каталог этого компьютера: интерфейс предложит перенести его на хаб.
    if ($core.legacy -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot "$($core.legacy)\catalog.sqlite"))) {
        $arguments += @('--legacy', (Join-Path $PSScriptRoot $core.legacy))
    }
    $env:PYTHONUTF8 = '1'
    & $python @arguments
    exit $LASTEXITCODE
}

if (!(Test-Path -LiteralPath (Join-Path $Catalog 'catalog.sqlite'))) {
    throw "Catalog not found: $Catalog"
}
& $python web_server.py --data $Catalog --host $ListenAddress --port $Port `
    --min-cluster-size $MinClusterSize --max-faces $MaxFaces `
    --token-file $TokenFile --no-browser `
    --device-id $DeviceId --device-name $DeviceName
