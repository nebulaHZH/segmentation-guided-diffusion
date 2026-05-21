param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("brisc_glioma_t1_gray","brisc_glioma_bac_t1_gray","cta", "ixi_t1_gray", "ixi_t2_gray")]
    [string]$DatasetName,

    [int]$ImageSize = 256,
    [int]$EvalBatchSize = 4,
    [int]$SampleSize = 200,
    [string]$ModelType = "DDIM",
    [ValidateSet("small", "base")]
    [string]$ModelSize = "small",
    [ValidateSet("DDPM", "DDIM")]
    [string]$SampleScheduler = "DDPM",
    [int]$NumInferenceSteps = 40
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$OutputDir = Join-Path $Repo ("outputs\" + $ModelType.ToLower() + "-" + $DatasetName + "-" + $ImageSize)

Push-Location $Repo
try {
    & $Python main.py `
        --mode eval_many `
        --model_type $ModelType `
        --model_size $ModelSize `
        --img_size $ImageSize `
        --num_img_channels 1 `
        --dataset $DatasetName `
        --eval_batch_size $EvalBatchSize `
        --eval_sample_size $SampleSize `
        --eval_scheduler $SampleScheduler `
        --eval_num_inference_steps $NumInferenceSteps `
        --output_dir $OutputDir
}
finally {
    Pop-Location
}
