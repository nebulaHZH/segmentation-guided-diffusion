# Anatomical Registers Module

This branch adds support for **Anatomical Register Modulation** to the segmentation-guided diffusion models. This feature incorporates learnable anatomical knowledge into the diffusion process through specialized register tokens.

## What are Anatomical Registers?

Anatomical registers are learnable tokens that encode different aspects of medical anatomy:
- **Organ Registers**: Encode organ-specific features (heart, lungs, liver, etc.)
- **Spatial Registers**: Encode anatomical directions (anterior, posterior, left, right, superior, inferior)
- **Scale Registers**: Encode different levels of detail (organ-level, tissue-level, cellular, fine details)

## How It Works

The anatomical registers modulate the UNet's behavior based on the diffusion timestep:
- **Early timesteps (t > 500)**: Emphasizes organ and spatial registers for layout generation
- **Late timesteps (t < 500)**: Emphasizes scale registers for detail refinement

## Usage

### Training with Anatomical Registers

To train a model with anatomical registers, add the `--use_anatomical_registers` flag:

```bash
python main.py \
    --mode train \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset chest_xray \
    --img_dir /path/to/images \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 200 \
    --use_anatomical_registers \
    --num_organ_registers 12 \
    --num_spatial_registers 6 \
    --num_scale_registers 4
```

### Training with Both Segmentation Guidance and Anatomical Registers

You can combine anatomical registers with segmentation guidance:

```bash
python main.py \
    --mode train \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset chest_xray \
    --img_dir /path/to/images \
    --seg_dir /path/to/masks \
    --segmentation_guided \
    --num_segmentation_classes 3 \
    --use_anatomical_registers \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 200
```

### Sampling from Trained Models

To generate samples from a model trained with anatomical registers:

```bash
python main.py \
    --mode eval_many \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset chest_xray \
    --use_anatomical_registers \
    --eval_batch_size 8 \
    --eval_sample_size 100
```

## Customization

You can adjust the number of registers for your specific use case:

- `--num_organ_registers`: Number of organ-specific registers (default: 12)
- `--num_spatial_registers`: Number of spatial registers (default: 6)
- `--num_scale_registers`: Number of scale registers (default: 4)

## Model Architecture

The anatomical registers are implemented as a wrapper around the base UNet2DModel:

1. **AnatomicalRegisterBank**: Manages the learnable register tokens
2. **RegisterModulatedUNet**: Wraps the UNet and applies register modulation
3. Stage-aware weighting based on diffusion timestep
4. Gated modulation to blend register features with UNet outputs

## Saved Model Format

When training with anatomical registers, the model saves two components:
- `unet/`: The base UNet model (standard diffusers format)
- `anatomical_registers.pt`: The register bank and modulation layers

Both files are automatically loaded when resuming training or during evaluation.

## Implementation Details

- Register dimension: 512 (matches UNet hidden dimension)
- Xavier initialization for register tokens
- Learnable mixing weights for register types
- Sigmoid gating for feature modulation
- Compatible with both DDPM and DDIM schedulers
- Works with or without segmentation guidance
- Supports multi-GPU training via DataParallel