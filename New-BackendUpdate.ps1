<#
  Собирает полный снимок исходников HomeCloud Core из Git-рабочей копии.
  В пакет попадают tracked-файлы и новые неигнорируемые файлы, поэтому
  связанные модули обновляются вместе и не расходятся по версиям.
#>
param(
    [string]$Output = (Join-Path $PSScriptRoot 'work\homecloud-core-update.zip')
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (!(Get-Command git -ErrorAction SilentlyContinue)) { throw 'git не найден' }
if (!(Get-Command tar.exe -ErrorAction SilentlyContinue)) { throw 'tar.exe не найден' }

$files = @(git ls-files --cached --others --exclude-standard) |
    Where-Object { $_ -and $_ -notmatch '^(?:work|data|models|downloads|trash-catalog|trash-clean-catalog)/' }
if (!$files.Count) { throw 'В рабочей копии нет файлов для упаковки' }

$outputPath = [IO.Path]::GetFullPath($Output)
$outputDir = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
if (Test-Path -LiteralPath $outputPath) { Remove-Item -LiteralPath $outputPath -Force }

& tar.exe -a -c -f $outputPath -C $PSScriptRoot @files
if ($LASTEXITCODE -ne 0) { throw "tar.exe завершился с кодом $LASTEXITCODE" }

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash
Write-Output ([pscustomobject]@{
    Package = $outputPath
    Files = $files.Count
    Sha256 = $hash
})
