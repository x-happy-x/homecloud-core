$ErrorActionPreference = 'Stop'
$prototype = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Resolve-Path (Join-Path $prototype '.venv\Scripts\python.exe')).Path
$ocrPython = 'C:\cv-ocr\Scripts\python.exe'
$catalog = Join-Path $prototype 'trash-clean-catalog'
$progress = Join-Path $catalog 'analysis-progress.json'
$stop = Join-Path $catalog 'analysis-stop.request'
$log = Join-Path $prototype 'trash-analysis.log'
$errorLog = Join-Path $prototype 'trash-analysis-error.log'
$env:HF_HOME = 'C:\cv-models\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PADDLE_PDX_CACHE_HOME = 'C:\cv-models\paddle'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'

Remove-Item -LiteralPath $stop -Force -ErrorAction SilentlyContinue
"[$(Get-Date -Format o)] Starting local visual analysis" | Add-Content -LiteralPath $log
& $python (Join-Path $prototype 'analyze_photos.py') analyze `
    --catalog $catalog --limit 250 --batch-size 16 `
    --progress-file $progress --stop-file $stop 1>> $log 2>> $errorLog
$exitCode = $LASTEXITCODE
if ($exitCode -eq 0 -and !(Test-Path -LiteralPath $stop)) {
    & $ocrPython (Join-Path $prototype 'ocr_photos.py') `
        --catalog $catalog --limit 250 --progress-file $progress --stop-file $stop `
        1>> $log 2>> $errorLog
    $exitCode = $LASTEXITCODE
}
if ($exitCode -eq 0 -and !(Test-Path -LiteralPath $stop)) {
    & $python (Join-Path $prototype 'caption_photos.py') `
        --catalog $catalog --limit 25 --progress-file $progress --stop-file $stop `
        1>> $log 2>> $errorLog
    $exitCode = $LASTEXITCODE
}
"[$(Get-Date -Format o)] Analysis finished with exit code $exitCode" | Add-Content -LiteralPath $log
exit $exitCode
