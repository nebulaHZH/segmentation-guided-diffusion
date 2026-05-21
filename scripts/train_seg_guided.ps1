param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("brisc_glioma_t1_gray", "brisc_glioma_bac_t1_gray")]
    [string]$DatasetName,

    [string]$DataRoot = "",
    [int]$ImageSize = 256,
    [int]$TrainBatchSize = 2,
    [int]$EvalBatchSize = 4,
    [int]$NumEpochs = 200,
    [double]$LearningRate = 2e-5,
    [string]$ModelType = "DDIM",
    [ValidateSet("small", "base")]
    [string]$ModelSize = "small",
    [int]$NumSegmentationClasses = 2,
    [int]$SaveImageEpochs = 10,
    [int]$SaveModelEpochs = 25,
    [Alias("resume_epoch")]
    [int]$ResumeEpoch = -1,
    [switch]$NoResume,
    [switch]$UseAblatedSegmentations,
    [switch]$RebuildSegDir
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
$MaskDir = Join-Path $DataRoot ($DatasetName + "_mask")
$SegDir = Join-Path $DataRoot ($DatasetName + "_seg")
$SegType = "all"
$OutputDir = Join-Path $Repo ("outputs\" + $ModelType.ToLower() + "-" + $DatasetName + "-" + $ImageSize)

if (-not (Test-Path $ImgDir)) {
    throw "Dataset folder not found: $ImgDir. Run scripts\prepare_my_datasets.ps1 first."
}

if (-not (Test-Path $MaskDir)) {
    throw "Mask folder not found: $MaskDir. Create matching masks before segmentation-guided training."
}

foreach ($Split in @("train", "val", "test")) {
    $ImageSplitDir = Join-Path $ImgDir $Split
    $MaskSplitDir = Join-Path $MaskDir $Split
    $SegSplitDir = Join-Path (Join-Path $SegDir $SegType) $Split

    if (-not (Test-Path $ImageSplitDir)) {
        throw "Image split folder not found: $ImageSplitDir"
    }

    if (-not (Test-Path $MaskSplitDir)) {
        throw "Mask split folder not found: $MaskSplitDir"
    }

    New-Item -ItemType Directory -Force -Path $SegSplitDir | Out-Null

    $ImageFiles = Get-ChildItem -LiteralPath $ImageSplitDir -File | Sort-Object Name
    $Copied = 0
    $Existing = 0
    $Missing = New-Object System.Collections.Generic.List[string]

    foreach ($ImageFile in $ImageFiles) {
        $MaskPath = Join-Path $MaskSplitDir $ImageFile.Name
        if (-not (Test-Path -LiteralPath $MaskPath)) {
            $Missing.Add($ImageFile.Name) | Out-Null
            continue
        }

        $DestPath = Join-Path $SegSplitDir $ImageFile.Name
        $MaskFile = Get-Item -LiteralPath $MaskPath
        $ShouldCopy = $RebuildSegDir -or (-not (Test-Path -LiteralPath $DestPath))

        if (-not $ShouldCopy) {
            $DestFile = Get-Item -LiteralPath $DestPath
            $ShouldCopy = ($DestFile.Length -ne $MaskFile.Length) -or ($DestFile.LastWriteTimeUtc -lt $MaskFile.LastWriteTimeUtc)
        }

        if ($ShouldCopy) {
            Copy-Item -LiteralPath $MaskPath -Destination $DestPath -Force
            $Copied++
        }
        else {
            $Existing++
        }
    }

    if ($Missing.Count -gt 0) {
        $Preview = ($Missing | Select-Object -First 10) -join ", "
        throw "Missing $($Missing.Count) masks in $MaskSplitDir for split '$Split'. First missing files: $Preview"
    }

    $SegCount = (Get-ChildItem -LiteralPath $SegSplitDir -File).Count
    if ($SegCount -ne $ImageFiles.Count) {
        throw "Mask count mismatch for split '$Split': images=$($ImageFiles.Count), masks=$SegCount"
    }

    Write-Host ("{0}: images={1}, masks={2}, copied={3}, existing={4}" -f $Split, $ImageFiles.Count, $SegCount, $Copied, $Existing)
}

$MainArgs = @(
    "main.py",
    "--mode", "train",
    "--model_type", $ModelType,
    "--model_size", $ModelSize,
    "--img_size", $ImageSize,
    "--num_img_channels", "1",
    "--dataset", $DatasetName,
    "--img_dir", $ImgDir,
    "--seg_dir", $SegDir,
    "--segmentation_guided",
    "--segmentation_channel_mode", "single",
    "--num_segmentation_classes", $NumSegmentationClasses,
    "--train_batch_size", $TrainBatchSize,
    "--eval_batch_size", $EvalBatchSize,
    "--num_epochs", $NumEpochs,
    "--learning_rate", $LearningRate,
    "--save_image_epochs", $SaveImageEpochs,
    "--save_model_epochs", $SaveModelEpochs,
    "--output_dir", $OutputDir
)

if ($ResumeEpoch -ge 0) {
    $MainArgs += @("--resume_epoch", $ResumeEpoch)
}
elseif (-not $NoResume) {
    $MainArgs += "--resume_latest"
}

if ($UseAblatedSegmentations) {
    $MainArgs += "--use_ablated_segmentations"
}

Write-Host "Segmentation dir: $SegDir"
Write-Host "Output dir base: $OutputDir"
Write-Host "Note: main.py appends '-segguided' to the final output directory."
if ($ResumeEpoch -ge 0) {
    Write-Host "Resume: loading model weights and starting at epoch $ResumeEpoch."
}
elseif ($NoResume) {
    Write-Host "Resume: disabled; starting a fresh training run."
}
else {
    Write-Host "Resume: automatic; will use training_state.pt if it exists."
}

Push-Location $Repo
try {
    & $Python @MainArgs
}
finally {
    Pop-Location
}
