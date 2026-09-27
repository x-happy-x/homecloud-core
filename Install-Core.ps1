<#
  Ставит или обновляет ядро HomeCloud из пакета, который прислал хаб.

  Вызывает хаб по SSH (кнопки «Установить», «Обновить», «Перевести на хаб»
  в разделе «Сканирование»). Порядок:
    1. раскладывает пакет поверх папки ядра, прежние версии файлов — в
       core-data\backups\<время>;
    2. пишет core.json (кто это ядро), core-data\hub.json (адрес и токен связи
       с хабом) и backend-token.txt (токен, которым хаб подписывает запросы);
    3. при установке — окружение Python (setup.ps1) и автозапуск при входе;
       при обновлении — только недостающие лёгкие пакеты;
    4. перезапускает службу ядра через start-remote.ps1.
#>
param(
    [Parameter(Mandatory)] [string]$Package,
    [Parameter(Mandatory)] [string]$Repository,
    [ValidateSet('install', 'update')] [string]$Mode = 'update',
    [Parameter(Mandatory)] [string]$HubUrl,
    [Parameter(Mandatory)] [string]$CoreId,
    [string]$CoreName = $CoreId,
    [Parameter(Mandatory)] [string]$LinkToken,
    [Parameter(Mandatory)] [string]$CoreToken,
    [int]$Port = 18311
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = [IO.Path]::GetFullPath($Repository)
$packagePath = [IO.Path]::GetFullPath($Package)
if (!(Test-Path -LiteralPath $packagePath -PathType Leaf)) { throw "Пакет не найден: $packagePath" }
New-Item -ItemType Directory -Path $repo -Force | Out-Null

function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}

Write-Output "Ядро $CoreId, пакет $(Split-Path -Leaf $packagePath), режим $Mode"

# 1. Код.
$stage = Join-Path $env:TEMP ('homecloud-core-' + [guid]::NewGuid().ToString('N'))
# Не в work: там бывает ссылка на другой диск, а сеанс SSH через неё не ходит.
$backup = Join-Path $repo ('core-data\backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $stage, $backup -Force | Out-Null
try {
    Expand-Archive -LiteralPath $packagePath -DestinationPath $stage -Force
    $incoming = @(Get-ChildItem -LiteralPath $stage -File -Recurse)
    foreach ($file in $incoming) {
        $relative = $file.FullName.Substring($stage.Length).TrimStart('\')
        $target = Join-Path $repo $relative
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $saved = Join-Path $backup $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
            Copy-Item -LiteralPath $target -Destination $saved -Force
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target -Force
    }
    Write-Output "Файлов разложено: $($incoming.Count); прежние версии — $backup"
} finally {
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
}

# 2. Кто это ядро и где хаб.
$data = Join-Path $repo 'core-data'
New-Item -ItemType Directory -Path $data -Force | Out-Null
$legacy = if (Test-Path -LiteralPath (Join-Path $repo 'trash-clean-catalog\catalog.sqlite')) { 'trash-clean-catalog' } else { '' }
Write-Utf8 (Join-Path $repo 'core.json') (@{id = $CoreId; name = $CoreName; legacy = $legacy} | ConvertTo-Json)
Write-Utf8 (Join-Path $data 'hub.json') (@{url = $HubUrl; token = $LinkToken; core = $CoreId} | ConvertTo-Json)
Write-Utf8 (Join-Path $repo 'backend-token.txt') ($CoreToken + "`n")
Write-Output "Связь с хабом: $HubUrl"

# 3. Окружение.
# Windows PowerShell 5.1 при Stop делает исключение из любой строки stderr
# внешней программы (предупреждения pip, трейсбек проверки импорта), даже при
# 2>$null. Итог таких команд смотрим только по $LASTEXITCODE.
function Invoke-Native([scriptblock]$Block) {
    $ErrorActionPreference = 'Continue'
    & $Block 2>&1 | ForEach-Object { "$_" }
}
$python = Join-Path $repo '.venv\Scripts\python.exe'
if ($Mode -eq 'install' -or !(Test-Path -LiteralPath $python)) {
    Write-Output 'Ставлю окружение Python (setup.ps1) — это надолго'
    & (Join-Path $repo 'setup.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'setup.ps1 завершился с ошибкой' }
}
# Лёгкие пакеты ядра при хабе: SMB и SSH для источников.
Invoke-Native { & $python -c "import paramiko, smbclient" } | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output 'Доустанавливаю paramiko и smbprotocol'
    $packages = { & $python -m pip install --disable-pip-version-check -q 'paramiko>=3.4,<5' 'smbprotocol>=1.13,<2' }
    Invoke-Native $packages
    if ($LASTEXITCODE -ne 0) {
        # Бывает, что pip в окружении недообновлён и сам не запускается
        # (ImportError внутри pip). Ставим его заново из колеса, которое
        # лежит в самом Python, — это без сети — и пробуем ещё раз.
        $base = (& $python -c "import sys; print(sys.base_prefix)" | Select-Object -First 1)
        $wheel = Get-ChildItem -Path (Join-Path $base 'Lib\ensurepip\_bundled') -Filter 'pip-*.whl' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (!$wheel) { throw 'Не удалось поставить paramiko и smbprotocol, а колеса pip для починки нет' }
        Write-Output "Чиню pip в окружении из $($wheel.Name)"
        # Колесо в PYTHONPATH идёт раньше сломанного pip окружения; запуск
        # именно через -m pip, иначе pip отказывается менять сам себя.
        $env:PYTHONPATH = $wheel.FullName
        try {
            Invoke-Native { & $python -m pip install --disable-pip-version-check -q --force-reinstall $wheel.FullName }
        } finally {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        }
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось починить pip в окружении' }
        Invoke-Native $packages
    }
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось поставить paramiko и smbprotocol' }
}
$files = @(Get-ChildItem -LiteralPath $repo -Filter '*.py' -File | Select-Object -ExpandProperty FullName)
Invoke-Native { & $python -m py_compile @files }
if ($LASTEXITCODE -ne 0) { throw 'Проверка Python завершилась с ошибкой' }

if ($Mode -eq 'install') {
    # Автозапуск при входе в Windows; без прав администратора — только предупреждение.
    try {
        $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f (Join-Path $repo 'start-remote.ps1')) `
            -WorkingDirectory $repo
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
        Register-ScheduledTask -TaskName 'HomeCloud Core' -Action $action -Trigger $trigger -Force | Out-Null
        Write-Output 'Автозапуск: задача «HomeCloud Core» при входе'
    } catch {
        Write-Warning "Автозапуск не настроен: $($_.Exception.Message)"
    }
}

# 4. Перезапуск службы ядра.
$client = New-Object Net.Sockets.TcpClient
try {
    $busy = $client.BeginConnect('127.0.0.1', $Port, $null, $null).AsyncWaitHandle.WaitOne(500) -and $client.Connected
} finally { $client.Close() }
if ($busy) {
    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        # Вместе со службой уходят и её задания со всеми потомками: этап
        # (prototype.py и др.) — внук службы и иначе продолжал бы работать
        # рядом с новым заданием, деля с ним файл прогресса.
        $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
        $tree = @($listener.OwningProcess)
        for ($i = 0; $i -lt $tree.Count; $i++) {
            $tree += @($all | Where-Object { $_.ParentProcessId -eq $tree[$i] -and $tree -notcontains $_.ProcessId } |
                ForEach-Object { $_.ProcessId })
        }
        [array]::Reverse($tree)
        foreach ($id in $tree) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds 1
}
& (Join-Path $repo 'start-remote.ps1') -Port $Port
$deadline = (Get-Date).AddSeconds(60)
do {
    Start-Sleep -Milliseconds 700
    $client = New-Object Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $Port); $ready = $true } catch { $ready = $false } finally { $client.Dispose() }
} until ($ready -or (Get-Date) -ge $deadline)
if (!$ready) { throw "Ядро не открыло порт $Port — см. backend.error.log" }
Write-Output "Ядро запущено на порту $Port"
