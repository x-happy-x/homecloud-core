$ErrorActionPreference = 'Stop'

$prototype = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Resolve-Path (Join-Path $prototype '..\..\work\v\Scripts\python.exe')).Path
$catalog = Join-Path $prototype 'trash-clean-catalog'
$models = Join-Path $prototype 'models\buffalo_l'
$log = Join-Path $prototype 'trash-clean-scan.log'
$errorLog = Join-Path $prototype 'trash-clean-scan-error.log'
$progress = Join-Path $catalog 'scan-progress.json'
$stop = Join-Path $catalog 'scan-stop.request'

$excludedNames = @(
    '.git', '.gradle', '.idea', '.svn', '.vs', '.venv', '__pycache__',
    'AppData', 'bin', 'cache', 'caches', 'node_modules', 'obj', 'packages',
    'Program Files', 'Program Files (x86)', 'Projects', 'OldProjects',
    'site-packages', 'temp', 'tmp', 'venv', 'Virtual Machines', 'Windows',
    '3ds Max 2023', '3dsmax', 'Actual Installer', 'Arduino', 'Code Snippets',
    'Configs', 'desktop_webview_window', 'EasyTune', 'GitHub', 'IISExpress',
    'Image-Line', 'InfoBase', 'MATLAB', 'My Web Sites', 'PortableGit',
    'PowerShell', 'PowerToys', 'Rainmeter', 'SQL Server Management Studio',
    'Unreal Projects', 'Visual Studio 2017', 'Visual Studio 2019',
    'Visual Studio 2022', 'WindowsPowerShell',
    'Arma 3', "Assassin's Creed III", "Assassin's Creed Odyssey",
    "Assassin's Creed Origins", "Assassin's Creed Valhalla",
    'Assetto Corsa Competizione', 'Atomic Heart', 'Call of Duty',
    'CD Projekt Red', 'DARKSiDERS', 'Diablo II', 'DyingLight', 'Eek',
    'Electronic Arts', 'Fortnite', 'Game', 'GameCenter', 'Genshin Impact',
    'Ghost Games', 'Grand Theft Auto V', 'Hitman 3', 'Hogwarts Legacy',
    'Horizon Zero Dawn', 'Klei', 'Life Is Strange by xatab',
    "Marvel's Spider-Man Miles Morales", "Marvel's Spider-Man Remastered",
    'Metro Exodus', 'Mortal Kombat 11', 'My Games', 'Need for Speed Heat',
    'Need for Speed(TM) Payback', 'Paradox Interactive', 'Phasmophobia',
    'Rockstar Games', 'Ryujinx', 'Saints Row', 'SandRock', 'Sims 4 Studio',
    'Square Enix', 'steamvr', 'Teardown', 'The Witcher 3', 'UbisoftConnect',
    'Universe Sandbox', 'VRChat', 'Warface', 'Watch Dogs Legion', 'Игра'
)

$scanArgs = @(
    (Join-Path $prototype 'prototype.py'), 'scan',
    '--photos', 'D:\trash',
    '--models', $models,
    '--data', $catalog,
    '--limit', '1000000',
    '--min-side', '160',
    '--exclude-path', 'D:\trash\game data',
    '--exclude-file-pattern', '*__an__*',
    '--progress-file', $progress,
    '--stop-file', $stop
)
foreach ($name in $excludedNames) {
    $scanArgs += @('--exclude-dir-name', $name)
}

"[$(Get-Date -Format o)] Starting clean D:\trash scan" | Add-Content -LiteralPath $log
"[$(Get-Date -Format o)] Starting clean D:\trash scan" | Add-Content -LiteralPath $errorLog
Remove-Item -LiteralPath $stop -Force -ErrorAction SilentlyContinue
& $python @scanArgs 1>> $log 2>> $errorLog
$exitCode = $LASTEXITCODE
"[$(Get-Date -Format o)] Scan finished with exit code $exitCode" | Add-Content -LiteralPath $log
exit $exitCode
