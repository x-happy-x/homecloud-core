param(
    [string]$Catalog = (Join-Path $PSScriptRoot 'trash-catalog'),
    [int]$MinClusterSize = 3
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Run setup.ps1 first' }
if (!(Test-Path -LiteralPath (Join-Path $Catalog 'catalog.sqlite'))) {
    throw "Catalog not found: $Catalog"
}
& $python people_gui.py --data $Catalog --min-cluster-size $MinClusterSize
