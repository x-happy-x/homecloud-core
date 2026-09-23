<#
  Публикует homecloud-core на сервер HomeCloud (VM family-apps).

  Одно и то же ядро кода живёт в двух ролях:
    * хаб — контейнер homecloud-hub в /opt/homeapps: каталог, превью, реестры;
    * ядра — PC-X, PC-A: их ставят и обновляют из интерфейса пакетом с хаба.

  Скрипт:
    1. гоняет быстрые тесты;
    2. собирает пакет ядра (исходники из git + VERSION.json) и кладёт его в
       /srv/homeapps/homecloud/core-packages — после этого в интерфейсе у ядер
       появляется «Обновить»;
    3. копирует исходники хаба в /opt/homeapps/homecloud-core и пересобирает
       контейнер homecloud-hub.
#>
param(
    [string]$Target       = 'amagomedsharipov@192.168.99.20',
    [string]$RemotePath   = '/opt/homeapps/homecloud-core',
    [string]$ComposePath  = '/opt/homeapps',
    [string]$DataPath     = '/srv/homeapps/homecloud',
    [string]$IdentityFile = (Join-Path $env:USERPROFILE '.ssh\id_ed25519_pve'),
    [switch]$SkipTests,
    [switch]$SkipRestart
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$ssh = @('-i', $IdentityFile, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new')
function Invoke-Remote([string]$Command) {
    & ssh.exe @ssh $Target $Command
    if ($LASTEXITCODE -ne 0) { throw "Удалённая команда завершилась с кодом ${LASTEXITCODE}: $Command" }
}

if (!(Test-Path -LiteralPath $IdentityFile)) { throw "Нет ключа $IdentityFile" }
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (!$SkipTests) {
    & $python -m unittest test_pathkeys test_hub test_envs
    if ($LASTEXITCODE -ne 0) { throw 'Тесты не прошли' }
}

# Версия: дата публикации и коммит; незакоммиченные правки видны по «-dirty».
$commit = (& git describe --always --dirty).Trim()
$version = (Get-Date -Format 'yyyy.MM.dd-HHmm') + '-' + $commit
$work = Join-Path $PSScriptRoot 'work\packages'
New-Item -ItemType Directory -Path $work -Force | Out-Null
$stamp = Join-Path $env:TEMP ('homecloud-version-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stamp -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $stamp 'VERSION.json'),
    (@{version = $version; built_at = (Get-Date).ToString('o')} | ConvertTo-Json),
    (New-Object Text.UTF8Encoding($false)))

$files = @(git ls-files --cached --others --exclude-standard) |
    Where-Object { $_ -and $_ -notmatch '^(?:work|data|models|downloads|trash-catalog|trash-clean-catalog|test-photos)/' } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
$package = Join-Path $work "homecloud-core-$version.zip"
& tar.exe -a -c -f $package -C $PSScriptRoot @files -C $stamp VERSION.json
if ($LASTEXITCODE -ne 0) { throw "tar.exe завершился с кодом $LASTEXITCODE" }
Remove-Item -LiteralPath $stamp -Recurse -Force
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $package).Hash.ToLowerInvariant()
Write-Host "Пакет ядра ${version}: $($files.Count) файлов"

Write-Host "-> $Target`:$RemotePath"
Invoke-Remote "mkdir -p $RemotePath && rm -f $RemotePath/*.py"
$hubFiles = @(Get-ChildItem -LiteralPath $PSScriptRoot -Filter '*.py' -File |
    Where-Object { $_.Name -notlike 'test_*' } | Select-Object -ExpandProperty Name) +
    @('Dockerfile.hub', 'requirements-hub.txt')
& scp.exe @ssh -q @hubFiles "${Target}:$RemotePath/"
if ($LASTEXITCODE -ne 0) { throw 'scp исходников хаба не удался' }

$name = Split-Path -Leaf $package
& scp.exe @ssh -q $package "${Target}:/tmp/$name"
if ($LASTEXITCODE -ne 0) { throw 'scp пакета не удался' }
$current = (@{version = $version; file = $name; sha256 = $sha; built_at = (Get-Date).ToString('o')} |
    ConvertTo-Json -Compress).Replace("'", "")
Invoke-Remote ("sudo mkdir -p $DataPath/core-packages && sudo mv /tmp/$name $DataPath/core-packages/ " +
    "&& echo '$current' | sudo tee $DataPath/core-packages/current.json >/dev/null " +
    "&& ls -1t $DataPath/core-packages/homecloud-core-*.zip | tail -n +6 | sudo xargs -r rm -f")

if ($SkipRestart) { Write-Host 'Пакет и исходники на месте, хаб не перезапускался.'; return }
Invoke-Remote "cd $ComposePath && sudo docker compose up -d --build homecloud-hub"
Invoke-Remote 'for i in $(seq 1 60); do curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:18401/config | grep -q 403 && exit 0; sleep 2; done; exit 1'
Write-Host "Готово: хаб поднят, пакет ядра $version опубликован."
