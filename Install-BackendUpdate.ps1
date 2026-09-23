<#
  Устанавливает полный пакет исходников, проверяет Python и перезапускает backend.
  Перед заменой сохраняет текущие версии затрагиваемых файлов в work\backups.
#>
param(
    [Parameter(Mandatory)] [string]$Package,
    [string]$Repository = $PSScriptRoot,
    [int]$Port = 18311
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($Repository)
$packagePath = [IO.Path]::GetFullPath($Package)
if (!(Test-Path -LiteralPath $packagePath -PathType Leaf)) { throw "Пакет не найден: $packagePath" }
if (!(Test-Path -LiteralPath $repo -PathType Container)) { throw "Каталог не найден: $repo" }

$stage = Join-Path $env:TEMP ('homecloud-update-' + [guid]::NewGuid().ToString('N'))
$backup = Join-Path $repo ('work\backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $stage,$backup -Force | Out-Null

try {
    Expand-Archive -LiteralPath $packagePath -DestinationPath $stage -Force
    $incoming = @(Get-ChildItem -LiteralPath $stage -File -Recurse)
    foreach ($file in $incoming) {
        $relative = [IO.Path]::GetRelativePath($stage, $file.FullName)
        $target = Join-Path $repo $relative
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $saved = Join-Path $backup $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
            Copy-Item -LiteralPath $target -Destination $saved -Force
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
    }

    $python = Join-Path $repo '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $python)) { throw "Python venv не найден: $python" }
    $pythonFiles = @(Get-ChildItem -LiteralPath $repo -Filter '*.py' -File | Select-Object -ExpandProperty FullName)
    & $python -m py_compile @pythonFiles
    if ($LASTEXITCODE -ne 0) { throw 'Проверка Python завершилась с ошибкой' }

    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 700
    & (Join-Path $repo 'start-remote.ps1') -Port $Port

    $deadline = (Get-Date).AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 500
        $client = [Net.Sockets.TcpClient]::new()
        try { $client.Connect('127.0.0.1', $Port); $ready = $true } catch { $ready = $false } finally { $client.Dispose() }
    } until ($ready -or (Get-Date) -ge $deadline)
    if (!$ready) { throw "Backend не открыл порт $Port после обновления" }

    Write-Output ([pscustomobject]@{Ok=$true; Files=$incoming.Count; Backup=$backup; Port=$Port})
} finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
