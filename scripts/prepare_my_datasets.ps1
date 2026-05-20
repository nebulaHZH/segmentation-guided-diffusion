param(
    [string]$OutputRoot = "",
    [int]$ImageSize = 256,
    [int]$Seed = 42,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $Repo "data\prepared"
}
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$overwriteFlag = @()
if ($Overwrite) {
    $overwriteFlag = @("--overwrite")
}

& $Python (Join-Path $Repo "scripts\prepare_2d_dataset.py") `
    --source "E:\0_nebula\data\cj_bad_resized_2" `
    --output (Join-Path $OutputRoot "cta") `
    --mode L `
    --size $ImageSize `
    --seed $Seed `
    @overwriteFlag

& $Python (Join-Path $Repo "scripts\prepare_2d_dataset.py") `
    --source "D:\0-nebula\dataset\ixi_paried\t2_resized" `
    --output (Join-Path $OutputRoot "ixi_t2_gray") `
    --mode L `
    --size $ImageSize `
    --seed $Seed `
    @overwriteFlag

& $Python (Join-Path $Repo "scripts\prepare_2d_dataset.py") `
    --source "D:\0-nebula\dataset\ixi_paried\t1_resized" `
    --output (Join-Path $OutputRoot "ixi_t1_gray") `
    --mode L `
    --size $ImageSize `
    --seed $Seed `
    @overwriteFlag
