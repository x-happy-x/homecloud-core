<#
  Поднимает backend.ps1 в фоне по команде SSH. Через WMI (Win32_Process.Create),
  а не Start-Process: Windows-OpenSSH держит всю SSH-сессию в одном Job Object и
  убивает его целиком при закрытии соединения — «отвязанный» через Start-Process
  процесс на самом деле остаётся его ребёнком и умирает вместе с сессией. Процесс,
  созданный через WMI, этому Job Object не подчиняется и переживает отключение.
  Перенаправление вывода в backend.log/backend.error.log делает cmd.exe: сам
  Win32_Process.Create ничего не знает про редирект.
  Ничего не делает, если порт уже занят — backend и так поднят.
#>
param(
    [int]$Port = 18311
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Не Get-NetTCPConnection: в SSH-сессии CIM/WMI-провайдер под ним иногда виснет
# на много минут вместо мгновенного ответа. Обычный TCP-коннект так не делает.
function Test-PortOpen([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        return $result.AsyncWaitHandle.WaitOne(500) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

if (Test-PortOpen -Port $Port) {
    Write-Output "Бэкенд уже слушает порт $Port"
    exit 0
}

$log = Join-Path $PSScriptRoot 'backend.log'
$errLog = Join-Path $PSScriptRoot 'backend.error.log'
$backend = Join-Path $PSScriptRoot 'backend.ps1'
$commandLine = 'cmd.exe /c powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" > "{1}" 2> "{2}"' -f $backend, $log, $errLog

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $commandLine
    CurrentDirectory = $PSScriptRoot
}
if ($result.ReturnValue -ne 0) {
    throw "Win32_Process.Create вернул код $($result.ReturnValue)"
}
Write-Output "Бэкенд запускается, pid $($result.ProcessId)"
