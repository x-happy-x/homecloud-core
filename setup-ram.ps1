# Окружение и веса RAM++ (Recognize Anything Plus) для вкладки «Обучение».
#
# Отдельное окружение не прихоть: RAM++ написан под transformers 4.25 и
# timm 0.4.12, а vision-venv живёт на свежих версиях. torch при этом не
# качается второй раз — work\ram-venv видит пакеты vision-venv через .pth-файл,
# но свои transformers/timm стоят раньше в пути поиска.
param(
    [string]$Python = 'R:\lang\python\3.10.6\python.exe',
    [string]$Models = 'C:\cv-models\ram-plus'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root 'work\ram-venv'
$vision = Join-Path $root 'work\vision-venv\lib\site-packages'
if (!(Test-Path -LiteralPath $vision)) { throw "Сначала нужно окружение vision-venv: $vision" }

if (!(Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
    & $Python -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'
& $py -m pip install --cache-dir (Join-Path $root 'work\pip-cache') `
    'transformers==4.25.1' 'tokenizers==0.13.3' 'huggingface_hub==0.20.3' 'scipy==1.15.3' 'numpy==2.2.6'
& $py -m pip install --no-deps --cache-dir (Join-Path $root 'work\pip-cache') 'timm==0.4.12' 'fairscale==0.4.4'
Set-Content -LiteralPath (Join-Path $venv 'lib\site-packages\zz-vision-venv.pth') -Value $vision -Encoding ASCII

$source = Join-Path $root 'work\recognize-anything'
if (!(Test-Path -LiteralPath $source)) {
    $zip = Join-Path $root 'work\recognize-anything.zip'
    Invoke-WebRequest -UseBasicParsing 'https://github.com/xinyu1205/recognize-anything/archive/refs/heads/main.zip' -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $root 'work') -Force
    Rename-Item -LiteralPath (Join-Path $root 'work\recognize-anything-main') -NewName 'recognize-anything'
    Remove-Item -LiteralPath $zip
}
& $py -m pip install --no-deps --no-build-isolation $source

New-Item -ItemType Directory -Force (Join-Path $Models 'bert-base-uncased') | Out-Null
$weights = Join-Path $Models 'ram_plus_swin_large_14m.pth'
if (!(Test-Path -LiteralPath $weights)) {
    # 3 ГБ; curl докачивает с места обрыва, если запустить скрипт ещё раз.
    curl.exe -L -C - --retry 20 -o "$weights.part" 'https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth'
    if ((Get-Item -LiteralPath "$weights.part").Length -ne 3010210801) { throw 'Веса RAM++ скачались не полностью — запустите скрипт ещё раз' }
    Move-Item -LiteralPath "$weights.part" -Destination $weights
}
foreach ($name in 'vocab.txt', 'tokenizer_config.json', 'tokenizer.json', 'config.json') {
    $target = Join-Path $Models "bert-base-uncased\$name"
    if (!(Test-Path -LiteralPath $target)) {
        curl.exe -L --retry 10 -o $target "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/$name"
    }
}
& $py -c "import torch, ram; print('RAM++ ready, CUDA:', torch.cuda.is_available())"
