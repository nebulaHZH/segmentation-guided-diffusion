param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("brisc_glioma_t1_gray", "brisc_glioma_bac_t1_gray")]
    [string]$DatasetName,

    [string]$DataRoot = "",
    [int]$ImageSize = 256,
    [int]$EvalBatchSize = 4,
    [int]$SampleSize = 32,
    [string]$ModelType = "DDIM",
    [ValidateSet("small", "base")]
    [string]$ModelSize = "small",
    [int]$NumSegmentationClasses = 2,
    [ValidateSet("DDPM", "DDIM")]
    [string]$SampleScheduler = "DDPM",
    [int]$NumInferenceSteps = 40,
    [ValidateSet("train", "val", "test")]
    [string]$MaskSplit = "train",
    [switch]$UseAblatedSegmentations
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

$SegDir = Join-Path $DataRoot ($DatasetName + "_seg")
$SegSplitDir = Join-Path (Join-Path $SegDir "all") $MaskSplit
$OutputDir = Join-Path $Repo ("outputs\" + $ModelType.ToLower() + "-" + $DatasetName + "-" + $ImageSize)
$FinalOutputDir = $OutputDir + "-segguided"

if (-not (Test-Path $SegSplitDir)) {
    throw "Segmentation $MaskSplit folder not found: $SegSplitDir. Run scripts\train_seg_guided.ps1 once or prepare *_seg\all\$MaskSplit masks."
}

if (-not (Test-Path $FinalOutputDir)) {
    throw "Segmentation-guided model output folder not found: $FinalOutputDir. Train the mask-guided model first."
}

$MaskCount = (Get-ChildItem -LiteralPath $SegSplitDir -File).Count
if ($MaskCount -le 0) {
    throw "No $MaskSplit masks found in: $SegSplitDir"
}

$MainArgs = @(
    "main.py",
    "--mode", "eval_many",
    "--model_type", $ModelType,
    "--model_size", $ModelSize,
    "--img_size", $ImageSize,
    "--num_img_channels", "1",
    "--dataset", $DatasetName,
    "--seg_dir", $SegDir,
    "--segmentation_guided",
    "--segmentation_channel_mode", "single",
    "--num_segmentation_classes", $NumSegmentationClasses,
    "--eval_batch_size", $EvalBatchSize,
    "--eval_sample_size", $SampleSize,
    "--eval_scheduler", $SampleScheduler,
    "--eval_num_inference_steps", $NumInferenceSteps,
    "--eval_split", $MaskSplit,
    "--output_dir", $OutputDir
)

if ($UseAblatedSegmentations) {
    $MainArgs += "--use_ablated_segmentations"
}

Write-Host "Segmentation dir: $SegDir"
Write-Host "Mask split: $MaskSplit ($MaskCount masks; masks will be reused if SampleSize is larger)"
Write-Host "Model dir: $FinalOutputDir"
Write-Host "Output samples: $(Join-Path $FinalOutputDir ("samples_many_" + $SampleSize + "_" + $MaskSplit))"

Push-Location $Repo
try {
    & $Python @MainArgs
}
finally {
    Pop-Location
}
