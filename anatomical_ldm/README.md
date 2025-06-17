# Anatomical Latent Diffusion Models

A complete implementation of anatomical-aware latent diffusion models for chest X-ray generation, addressing the fundamental issues identified in pixel-space approaches.

## 🎯 Key Features

- **Anatomical Cross-Attention**: Uses proven cross-attention mechanisms (like text conditioning) for anatomical guidance
- **Semantic Latent Space**: Operates in VAE latent space where anatomical concepts have natural representation  
- **Stage-Aware Conditioning**: Adapts anatomical guidance based on diffusion timestep (layout → details)
- **Multi-Scale Integration**: Anatomical conditioning at multiple UNet resolutions
- **Comprehensive Training**: Full pipelines for both VAE and LDM training
- **Extensive Evaluation**: Metrics for image quality, diversity, and anatomical consistency

## 🏗️ Architecture Overview

### Core Components

1. **AnatomicalVAE** - Extends `diffusers.AutoencoderKL` with anatomical consistency features
2. **AnatomicalRegisterBank** - Learnable organ-specific embeddings with spatial/temporal awareness
3. **AnatomicalUNet2DConditionModel** - Extends `diffusers.UNet2DConditionModel` with anatomical cross-attention
4. **AnatomicalLDMPipeline** - Complete inference pipeline with anatomical conditioning

### Why This Approach Works

**Problems with Previous Pixel-Space Approach:**
- ❌ Too late modulation (final output blend)
- ❌ Information destruction (spatial pooling)  
- ❌ Scale mismatch between features
- ❌ Wrong feature space (pixel vs semantic)

**Our LDM Solution:**
- ✅ Cross-attention conditioning (proven mechanism)
- ✅ Semantic latent space (natural anatomical representation)
- ✅ Multi-scale integration (conditioning throughout generation)
- ✅ Stage-aware adaptation (layout → detail progression)

## 🚀 Quick Start

### Installation

```bash
# Install requirements
pip install torch torchvision diffusers transformers accelerate
pip install torchmetrics wandb tqdm matplotlib seaborn
pip install datasets pillow numpy

# Optional: For anatomical evaluation
pip install scikit-learn opencv-python
```

### Training Pipeline

#### Stage 1: Train Anatomical VAE

```bash
python -m anatomical_ldm.train_vae \
    --train_dir /path/to/chest_xrays \
    --val_dir /path/to/val_xrays \
    --mask_dir /path/to/masks \  # Optional: for supervised anatomical loss
    --image_size 512 \
    --batch_size 16 \
    --num_epochs 200 \
    --output_dir outputs/vae \
    --use_wandb
```

#### Stage 2: Train Anatomical LDM

```bash
python -m anatomical_ldm.train_ldm \
    --train_dir /path/to/chest_xrays \
    --val_dir /path/to/val_xrays \
    --vae_path outputs/vae/vae_epoch_200 \
    --text_file /path/to/captions.json \  # Optional: for text conditioning
    --use_text_conditioning \
    --image_size 512 \
    --batch_size 16 \
    --num_epochs 1000 \
    --output_dir outputs/ldm \
    --use_wandb
```

### Evaluation

```bash
# Full evaluation with metrics
python -m anatomical_ldm.evaluate \
    --pipeline_path outputs/ldm/pipeline_epoch_1000 \
    --real_data_dir /path/to/real_xrays \
    --num_samples 1000 \
    --run_full_evaluation \
    --output_dir evaluation_results

# Generate samples only
python -m anatomical_ldm.evaluate \
    --pipeline_path outputs/ldm/pipeline_epoch_1000 \
    --num_samples 100 \
    --generate_samples_only
```

## 📊 Expected Results

Based on our architectural improvements, you should see:

**Image Quality:**
- **FID**: Significant improvement over pixel-space baseline
- **LPIPS**: Better perceptual similarity to real chest X-rays
- **Inception Score**: Higher quality and diversity

**Anatomical Quality:**
- Consistent organ positioning and size ratios
- Proper anatomical relationships (heart position, lung symmetry)
- Realistic bone and soft tissue contrast

**Controllability:**
- Ability to emphasize/de-emphasize anatomical structures
- Stage-aware conditioning (layout guidance early, details late)
- Text-guided generation (if using text conditioning)

## 🔬 Technical Details

### Anatomical Register Bank

The register bank learns organ-specific embeddings with:
- **Spatial Position Encoding**: 2D positional embeddings for anatomical layout
- **Timestep-Aware Modulation**: Different organ emphasis based on diffusion stage
- **Cross-Organ Interaction**: Transformer layers for anatomical relationships

### Cross-Attention Integration

Anatomical conditioning is injected via cross-attention layers:
```python
# Standard cross-attention for text
text_output = cross_attention(hidden_states, text_embeddings)

# Anatomical cross-attention (parallel)
anatomical_output = anatomical_attention(hidden_states, anatomical_embeddings)

# Gated combination
final_output = text_output * (1 - gate) + anatomical_output * gate
```

### Stage-Aware Conditioning

- **Early Stage (t=800-1000)**: Strong anatomical layout guidance
- **Middle Stage (t=200-800)**: Balanced conditioning  
- **Late Stage (t=0-200)**: Subtle anatomical detail refinement

## 📁 File Structure

```
anatomical_ldm/
├── __init__.py                 # Package exports
├── vae.py                     # AnatomicalVAE implementation
├── anatomical_registers.py    # Register bank and conditioning
├── anatomical_unet.py         # UNet with anatomical cross-attention
├── train_vae.py              # VAE training pipeline
├── train_ldm.py              # LDM training pipeline  
├── evaluate.py               # Evaluation and sampling utilities
└── README.md                 # This file
```

## 🎛️ Configuration Options

### VAE Training

- `--anatomical_loss_weight`: Weight for anatomical consistency loss (default: 0.1)
- `--beta_start`/`--beta_end`: KL divergence scheduling (default: 1e-6 to 1e-2)
- `--num_anatomical_regions`: Number of anatomical regions to predict (default: 12)

### LDM Training

- `--anatomical_conditioning_dim`: Dimension of anatomical embeddings (default: 512)
- `--num_organs`: Number of anatomical organs (default: 12)  
- `--unconditional_prob`: Probability of unconditional training for CFG (default: 0.1)
- `--use_ema`: Use exponential moving average for model weights (recommended)

### Evaluation

- `--guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `--num_inference_steps`: Number of denoising steps (default: 50)
- Various metrics: FID, LPIPS, Inception Score, Diversity

## 🔧 Troubleshooting

### Common Issues

1. **OOM during training**: Reduce batch size or image resolution
2. **Poor anatomical consistency**: Increase `--anatomical_loss_weight` during VAE training
3. **Gate collapse**: Use EMA and proper learning rate scheduling
4. **Text conditioning not working**: Ensure CLIP models are properly loaded

### Performance Tips

1. **Use mixed precision**: Add `--fp16` for faster training
2. **Multiple GPUs**: Use `torch.nn.DataParallel` or DDP
3. **Efficient evaluation**: Use smaller batch sizes for metric computation
4. **Checkpoint frequently**: Save every 10-50 epochs depending on dataset size

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@article{anatomical_ldm_2024,
  title={Anatomical Latent Diffusion Models for Medical Image Generation},
  author={[Authors]},
  journal={ICLR 2026},
  year={2024}
}
```

## 🤝 Contributing

1. Follow the established code structure
2. Add proper docstrings and type hints
3. Test with different dataset sizes and configurations
4. Update documentation for new features

## 📄 License

[Add appropriate license]

---

**Note**: This implementation addresses all fundamental issues identified in previous pixel-space approaches and should provide significant improvements in both image quality and anatomical consistency for chest X-ray generation.