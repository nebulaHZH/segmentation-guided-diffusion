# 在你的 2D 医学图像上训练

这个仓库已经配置好，可以在下面这些本地数据目录上做 2D 图像生成训练：

- `E:\0_nebula\dataset\brisc_sort\glioma\hor`
- `D:\0-nebula\dataset\ixi_paried\t2_resized`
- `D:\0-nebula\dataset\ixi_paried\t1_resized`

原始数据目录不会被修改。整理后的 `train`、`val`、`test` 数据会复制到
`data/prepared` 下面。

## 环境配置

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_venv.ps1
```

## 准备数据

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_my_datasets.ps1 -Overwrite
```

执行后会生成：

- `data/prepared/brisc_glioma_t1_gray`
- `data/prepared/ixi_t1_gray`
- `data/prepared/ixi_t2_gray`

这些数据都会被转换成单通道灰度 PNG，并划分为 `train`、`val`、`test`。

## 无 mask 训练

如果使用 8 GB RTX 4060 Laptop GPU，建议先用 small 模型和 batch size 2：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_unconditional.ps1 -DatasetName ixi_t2_gray
```

可用的数据集名称：

```text
brisc_glioma_t1_gray
brisc_glioma_bac_t1_gray
ixi_t1_gray
ixi_t2_gray
```

## 有 mask 训练

BRISC glioma 数据集的 mask 已经放在 `data/prepared/*_mask` 下时，可以执行
segmentation-guided 训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray
powershell -ExecutionPolicy Bypass -File .\scripts\train_seg_guided.ps1 -DatasetName brisc_glioma_bac_t1_gray
```

脚本会自动生成 `main.py` 需要的 mask 目录结构：

```text
data/prepared/{DATASET_NAME}_seg/all/train
data/prepared/{DATASET_NAME}_seg/all/val
data/prepared/{DATASET_NAME}_seg/all/test
```

默认输出目录为：

```text
outputs/ddim-{DATASET_NAME}-256-segguided
```

默认训练参数是 `DDIM`、`small` 模型、`train_batch_size=2`、
`num_segmentation_classes=2`。如果需要调整，可以在命令后追加参数，例如：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray -TrainBatchSize 1 -NumEpochs 300
```

## 无 mask 模型采样

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_unconditional.ps1 -DatasetName ixi_t2_gray -SampleSize 32
```

默认采样脚本使用 DDPM 采样器和 1000 个去噪步数。对于这个仓库里的 small
医学图像模型，这个设置通常更稳定。你也可以显式使用 DDIM：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_unconditional.ps1 -DatasetName ixi_t2_gray -SampleScheduler DDIM -NumInferenceSteps 50
```

生成结果会保存到对应的 `outputs/.../samples_many_*` 目录下。

## 有 mask 模型采样

带 mask 训练完成后，使用 `sample_seg_guided.ps1` 采样。脚本默认读取训练集
mask，即：

```text
data/prepared/{DATASET_NAME}_seg/all/train
```

同一张 mask 可以重复使用，因为每次采样的初始随机噪声不同，所以生成图像不会完全一样。
如果要生成 400 张图像，直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray -SampleSize 400
```

另一个 BRISC mask 数据集可以这样运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_seg_guided.ps1 -DatasetName brisc_glioma_bac_t1_gray -SampleSize 400
```

默认输出目录为：

```text
outputs/ddim-{DATASET_NAME}-256-segguided/samples_many_400_train
```

如果训练集 mask 数量少于 400，脚本会自动循环复用 mask，直到生成满 400 张。
生成文件名会带数字前缀，例如 `0000_condon_xxx.png`，因此重复使用同一张 mask
时不会覆盖之前生成的结果。

如果想改用验证集或测试集 mask，可以指定 `-MaskSplit`：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray -SampleSize 400 -MaskSplit val
powershell -ExecutionPolicy Bypass -File .\scripts\sample_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray -SampleSize 400 -MaskSplit test
```

默认采样器是 `DDPM`，去噪步数为 `40`。如果想用 DDIM，可以这样写：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sample_seg_guided.ps1 -DatasetName brisc_glioma_t1_gray -SampleSize 400 -SampleScheduler DDIM -NumInferenceSteps 50
```
