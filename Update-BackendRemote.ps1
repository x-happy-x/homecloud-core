<#
  Полное обновление удалённого Windows-backend через SSH.
  Пример:
    $env:HOMECLOUD_SSH_PASSWORD = '...'
    .\Update-BackendRemote.ps1 -HostName 192.168.1.20

  По умолчанию SSH вызывается напрямую. Для маршрута через Proxmox/контейнер
  передайте -SshCommand собственный скрипт-обёртку.
#>
param(
    [Parameter(Mandatory)] [string]$HostName,
    [string]$UserName = $env:USERNAME,
    [string]$RemoteRepository = 'P:\services\homecloud-core',
    [string]$IdentityFile = (Join-Path $env:USERPROFILE '.ssh\id_ed25519'),
    [string]$Password = $env:HOMECLOUD_SSH_PASSWORD
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$package = Join-Path $PSScriptRoot 'work\homecloud-core-update.zip'
& (Join-Path $PSScriptRoot 'New-BackendUpdate.ps1') -Output $package | Format-List

$target = "$UserName@$HostName"
$sshOptions = @('-o','StrictHostKeyChecking=accept-new')
if (Test-Path -LiteralPath $IdentityFile) { $sshOptions += @('-i',$IdentityFile) }

if ($Password) {
    if (!(Get-Command sshpass -ErrorAction SilentlyContinue)) {
        throw 'Для пароля нужен sshpass; лучше установить SSH-ключ или не передавать -Password'
    }
    $transport = @('sshpass','-p',$Password)
} else {
    $transport = @()
}

$remotePackage = 'homecloud-core-update.zip'
$remoteInstaller = 'Install-BackendUpdate.ps1'
if ($transport.Count) {
    & $transport[0] $transport[1..($transport.Count-1)] scp @sshOptions $package (Join-Path $PSScriptRoot 'Install-BackendUpdate.ps1') "${target}:"
} else {
    & scp @sshOptions $package (Join-Path $PSScriptRoot 'Install-BackendUpdate.ps1') "${target}:"
}
if ($LASTEXITCODE -ne 0) { throw 'Не удалось передать пакет' }

$escapedRepo = $RemoteRepository.Replace("'", "''")
$command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$remoteInstaller`" -Package `"$remotePackage`" -Repository `"$escapedRepo`""
if ($transport.Count) {
    & $transport[0] $transport[1..($transport.Count-1)] ssh @sshOptions $target $command
} else {
    & ssh @sshOptions $target $command
}
if ($LASTEXITCODE -ne 0) { throw 'Удалённая установка завершилась с ошибкой' }
