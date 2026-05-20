# Running On Your 2D Medical Images

This fork is configured for unconditional 2D image generation on these folders:

- `E:\0_nebula\dataset\brisc_sort\glioma\hor`
- `D:\0-nebula\dataset\ixi_paried\t2_resized`
- `D:\0-nebula\dataset\ixi_paried\t1_resized`

The source folders are left untouched. Prepared train/val/test copies are written
under `data/prepared`.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_venv.ps1
```

## Prepare Data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_my_datasets.ps1 -Overwrite
```

This creates:

- `data/prepared/brisc_glioma_t1_gray`
- `data/prepared/ixi_t1_gray`
- `data/prepared/ixi_t2_gray`

All three are converted to single-channel grayscale PNGs and split into
`train`, `val`, and `test`.

## Train

For an 8 GB RTX 4060 Laptop GPU, start with the small model and batch size 2:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_unconditional.ps1 -DatasetName ixi_t2_gray
```

Other valid dataset names:

```text
brisc_glioma_t1_gray
ixi_t1_gray
ixi_t2_gray
```

## Sample

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_unconditional.ps1 -DatasetName ixi_t2_gray -SampleSize 32
```

By default the sampling script uses a DDPM sampler with 1000 denoising steps,
which is more stable for the small 2D medical models in this fork. You can try
DDIM explicitly with `-SampleScheduler DDIM -NumInferenceSteps 50`.

Samples are saved under the corresponding `outputs/.../samples_many_*` folder.
