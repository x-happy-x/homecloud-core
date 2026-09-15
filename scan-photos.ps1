param(
    [Parameter(Mandatory = $true)]
    [string]$Photos,
    [string]$Catalog = (Join-Path $PSScriptRoot 'my-catalog'),
    [int]$Limit = 1000000,
    [ValidateSet('hdbscan', 'dbscan')]
    [string]$Algorithm = 'hdbscan',
    [int]$MinClusterSize = 3,
    [double]$Distance = 0.35
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Run setup.ps1 first' }
if (!(Test-Path -LiteralPath $Photos -PathType Container)) { throw "Photo folder not found: $Photos" }
$distanceArg = $Distance.ToString([Globalization.CultureInfo]::InvariantCulture)

& $python prototype.py scan --photos $Photos --models .\models\buffalo_l --data $Catalog --limit $Limit
if ($LASTEXITCODE -ne 0) { throw 'Scan failed' }
& $python prototype.py gallery --data $Catalog --algorithm $Algorithm --min-cluster-size $MinClusterSize --distance $distanceArg
if ($LASTEXITCODE -ne 0) { throw 'Gallery failed' }
Write-Host "Done. Open: $(Join-Path (Resolve-Path $Catalog) 'gallery.html')"
