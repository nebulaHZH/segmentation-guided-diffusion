param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("brisc_glioma_t1_gray","brisc_glioma_bac_t1_gray", "ixi_t1_gray", "ixi_t2_gray")]
    [string]$DatasetName,

    [string]$DataRoot = "",
    [int]$ImageSize = 256,
    [int]$TrainBatchSize = 2,
    [int]$EvalBatchSize = 4,
    [int]$NumEpochs = 200,
    [double]$LearningRate = 2e-5,
    [string]$ModelType = "DDIM",
    [ValidateSet("small", "base")]
    [string]$ModelSize = "small"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $Repo "data\prepared"
}
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ImgDir = Join-Path $DataRoot $DatasetName
$OutputDir = Join-Path $Repo ("outputs\" + $ModelType.ToLower() + "-" + $DatasetName + "-" + $ImageSize)

if (-not (Test-Path $ImgDir)) {
    throw "Dataset folder not found: $ImgDir. Run scripts\prepare_my_datasets.ps1 first."
}

Push-Location $Repo
try {
    & $Python main.py `
        --mode train `
        --model_type $ModelType `
        --model_size $ModelSize `
        --img_size $ImageSize `
        --num_img_channels 1 `
        --dataset $DatasetName `
        --img_dir $ImgDir `
        --train_batch_size $TrainBatchSize `
        --eval_batch_size $EvalBatchSize `
        --num_epochs $NumEpochs `
        --learning_rate $LearningRate `
        --save_image_epochs 10 `
        --save_model_epochs 25 `
        --output_dir $OutputDir
}
finally {
    Pop-Location
}
