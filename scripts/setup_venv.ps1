param(
    [string]$Python = "3.12"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $Repo
try {
    if (-not (Test-Path ".venv")) {
        uv venv .venv --python $Python
        & ".\.venv\Scripts\python.exe" -m ensurepip --upgrade
    }

    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu126
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt tensorboard safetensors "huggingface_hub==0.19.4" "pyarrow<15" "accelerate==0.24.1"
    & ".\.venv\Scripts\python.exe" -m pip install --force-reinstall --no-deps torch==2.11.0+cu126 torchvision==0.26.0+cu126 torchaudio==2.11.0+cu126 -f https://mirrors.aliyun.com/pytorch-wheels/cu126
    & ".\.venv\Scripts\python.exe" -m pip check
}
finally {
    Pop-Location
}
