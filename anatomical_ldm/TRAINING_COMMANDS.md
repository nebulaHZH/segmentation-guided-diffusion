# Training Commands for Your Dataset

## 📋 **Prerequisites**

1. **Dataset Structure**:
```
your_dataset/
├── train/
│   ├── images/          # Training images (.png, .jpg, .jpeg)
│   └── masks/           # Training masks (.png, .npy)
├── val/                 # Optional validation split
│   ├── images/
│   └── masks/
```

2. **Mask Format**: 
   - Pixel values = class indices: `0, 1, 2, 3, ...`
   - Class 0 = background
   - Classes 1+ = your anatomical structures

3. **Hardware Requirements**:
   - **Minimum**: 16GB GPU (RTX 3080/4080, V100)
   - **Recommended**: 24GB+ GPU (RTX 4090, A6000, A100)
   - **RAM**: 32GB+ system memory
   - **Storage**: 100GB+ free space

## 🚀 **Step 1: Install Dependencies**

```bash
# Clone the repository
git clone https://github.com/mazurowski-lab/segmentation-guided-diffusion.git
cd segmentation-guided-diffusion
git checkout anatomical-registers

# Install requirements
pip install -r anatomical_ldm/requirements.txt

# Install additional dependencies for training
pip install wandb accelerate
```

## 🔧 **Step 2: Configure Your Dataset**

Edit `anatomical_ldm/train_your_dataset.py` to match your dataset:

```python
# Modify lines 19-33 in train_your_dataset.py
YOUR_CLASSES = [
    "background",        # Class 0 (always background)
    "liver",            # Class 1 - Replace with your class names
    "kidney",           # Class 2
    "spleen",           # Class 3
    # ... Add your anatomical classes here
]
```

**Example configurations**:

### CT-org Dataset:
```python
YOUR_CLASSES = [
    "background", "liver", "bladder", "lungs", "kidneys", 
    "bone", "brain", "heart", "pancreas", "spleen", 
    "gallbladder", "esophagus", "stomach", "aorta"
]
```

### Brain MRI:
```python
YOUR_CLASSES = [
    "background", "cerebrospinal_fluid", "gray_matter", 
    "white_matter", "ventricles", "brainstem", "cerebellum", "skull"
]
```

### Chest X-ray:
```python
YOUR_CLASSES = [
    "background", "heart", "left_lung", "right_lung", 
    "liver", "ribs", "spine", "clavicle", "diaphragm"
]
```

## 🏗️ **Step 3: Train the VAE (Stage 1)**

```bash
# Train anatomical VAE first (2-4 days)
python -m anatomical_ldm.train_vae \
    --train_dir "path/to/your_dataset/train/images" \
    --val_dir "path/to/your_dataset/val/images" \
    --mask_dir "path/to/your_dataset/train/masks" \
    --val_mask_dir "path/to/your_dataset/val/masks" \
    --image_size 512 \
    --batch_size 16 \
    --num_epochs 200 \
    --num_anatomical_regions 12 \
    --anatomical_loss_weight 0.1 \
    --learning_rate 1e-4 \
    --output_dir "outputs/your_vae" \
    --use_wandb \
    --wandb_project "your-anatomical-vae"
```

**Adjust parameters**:
- `--num_anatomical_regions`: Set to `len(YOUR_CLASSES)`
- `--batch_size`: Reduce if OOM (try 8, 4, 2)
- `--image_size`: Reduce if OOM (try 256, 384)

**Monitor training**:
- Check WandB logs for reconstruction quality
- VAE should achieve good image reconstruction
- Wait for training to complete (~200 epochs)

## 🎯 **Step 4: Train the Supervised LDM (Stage 2)**

```bash
# Train the supervised LDM with anatomical registers (1-2 weeks)
python -m anatomical_ldm.train_your_dataset \
    --train_images "path/to/your_dataset/train/images" \
    --train_masks "path/to/your_dataset/train/masks" \
    --val_images "path/to/your_dataset/val/images" \
    --val_masks "path/to/your_dataset/val/masks" \
    --vae_path "outputs/your_vae/vae_epoch_200" \
    --image_size 512 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --stage1_epochs 100 \
    --stage1_weight 2.0 \
    --stage1_prob 1.0 \
    --stage2_epochs 200 \
    --stage2_weight 0.5 \
    --stage2_prob 0.6 \
    --stage3_epochs 100 \
    --stage3_weight 0.1 \
    --stage3_prob 0.3 \
    --output_dir "outputs/your_anatomical_ldm" \
    --save_every 20 \
    --validate_every 10 \
    --use_wandb \
    --wandb_project "your-anatomical-ldm" \
    --use_ema
```

## ⚙️ **Parameter Tuning Guide**

### **Memory Issues** (OOM):
```bash
# Reduce batch size
--batch_size 4        # Or even 2 for 16GB GPUs

# Reduce image size  
--image_size 256      # Or 384

# Reduce anatomical dimension
--anatomical_dim 256  # Instead of 512
```

### **Small Dataset** (<5k images):
```bash
# Reduce supervision to prevent overfitting
--stage1_weight 1.0   # Instead of 2.0
--stage2_weight 0.3   # Instead of 0.5
--stage3_weight 0.05  # Instead of 0.1

# Increase regularization
--supervision_probability 0.8  # Stage 1
--supervision_probability 0.4  # Stage 2  
--supervision_probability 0.2  # Stage 3
```

### **Large Dataset** (>50k images):
```bash
# Stronger supervision
--stage1_weight 3.0   # Even stronger
--stage1_epochs 150   # More epochs
--stage2_epochs 300   # More epochs

# More frequent validation
--validate_every 5
```

## 📊 **Monitor Training Progress**

### **Key Metrics to Watch**:
```bash
# In WandB, monitor:
- total_loss: Should decrease steadily
- diffusion_loss: Should decrease steadily  
- anatomical_supervision_loss: Should decrease over time
- used_supervision: Should be >0.5 in early stages
- val_loss: Should not increase (overfitting check)
```

### **Check Register Specialization**:
```python
# After ~50 epochs, check if registers are specializing
import torch
model = torch.load("outputs/your_anatomical_ldm/checkpoint_epoch_50.pt")
registers = model['unet_state_dict']['anatomical_registers.structure_embeddings']

# Compute similarities between registers
similarities = torch.cosine_similarity(registers[0:1], registers[1:])
print(f"Register similarities: {similarities}")

# Should see LOW similarities (< 0.7) indicating good specialization
```

## 🎮 **Test Your Trained Model**

```python
# test_generation.py
from anatomical_ldm.train_ldm import AnatomicalLDMPipeline
from torchvision.utils import save_image

# Load your trained model
pipeline = AnatomicalLDMPipeline.from_pretrained(
    "outputs/your_anatomical_ldm/final_model"
)

# Generate samples (no masks needed!)
print("Generating anatomically-aware samples...")
images = pipeline(
    prompt=None,  # Unconditional generation
    num_images_per_prompt=8,
    num_inference_steps=50,
    height=512,
    width=512,
)

# Save results
save_image(
    images, 
    "generated_anatomical_samples.png", 
    nrow=4, 
    normalize=True, 
    value_range=(0, 1)
)

print("✅ Generated samples saved to: generated_anatomical_samples.png")
print("Check that organs appear in anatomically correct locations!")
```

## 🚨 **Troubleshooting**

### **Problem**: OOM during training
```bash
# Solutions (try in order):
--batch_size 4
--image_size 384  
--anatomical_dim 256
# Use gradient checkpointing (add to trainer)
```

### **Problem**: Poor image quality
```bash
# Check VAE quality first:
python -c "
from anatomical_ldm.vae import AnatomicalVAE
vae = AnatomicalVAE.from_pretrained('outputs/your_vae/vae_epoch_200')
# Test reconstruction quality
"

# Reduce anatomical supervision if images look bad:
--stage1_weight 1.0
--stage2_weight 0.3
```

### **Problem**: Registers not specializing
```bash
# Increase supervision:
--stage1_weight 3.0
--stage1_prob 1.0  
--stage1_epochs 150

# Check mask loading:
# Ensure masks have correct class indices (0,1,2,...)
```

### **Problem**: Masks not loading
```bash
# Check mask format:
import numpy as np
mask = np.load("path/to/mask.npy")  # OR Image.open("mask.png")
print(f"Mask values: {np.unique(mask)}")
# Should see: [0, 1, 2, 3, ...] (class indices)

# Check file matching:
# Ensure mask filename matches image filename
```

## ⏱️ **Expected Timeline**

| Stage | Duration | GPU Memory | Purpose |
|-------|----------|------------|---------|
| VAE Training | 2-4 days | 12-16GB | Learn image encoding/decoding |
| LDM Stage 1 | 3-5 days | 16-20GB | Force register specialization |
| LDM Stage 2 | 1-2 weeks | 16-20GB | Balance diffusion + anatomy |
| LDM Stage 3 | 3-5 days | 16-20GB | Fine-tune with minimal supervision |

**Total: 3-4 weeks** on good hardware

## ✅ **Success Criteria**

Your training is successful if:

1. **VAE Stage**: Good image reconstruction quality
2. **LDM Stage 1**: Registers show low similarity (< 0.7)
3. **LDM Stage 2**: Stable training, good sample quality
4. **LDM Stage 3**: Generated images have consistent anatomical layout
5. **Final Test**: Organs appear in correct locations without mask input

**Ready to train on your dataset!** 🚀

---

## 📞 **Need Help?**

- Check WandB logs for training curves
- Ensure masks have correct format (class indices)
- Start with smaller image size (256px) for faster iteration
- Monitor GPU memory usage and adjust batch size accordingly