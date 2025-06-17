# Usage Guide: General Supervised Anatomical Registers

## ✅ Fixed Implementation Ready for Any Medical Dataset

The **fixed** implementation (`general_supervised_registers.py`) removes all chest X-ray specific assumptions and works with any multiclass medical dataset.

## 🎯 Key Features

✅ **Modality Agnostic**: Works with CT, MRI, X-ray, ultrasound, etc.  
✅ **Multiclass Support**: Each mask class automatically maps to one register  
✅ **General Spatial Learning**: Learns anatomical relationships from data  
✅ **No Hardcoded Priors**: Works without anatomical assumptions  

## 🚀 Quick Start

### Step 1: Prepare Your Dataset
```
your_dataset/
├── images/
│   ├── patient_001.png
│   ├── patient_002.png
│   └── ...
└── masks/
    ├── patient_001.png  # OR .npy files
    ├── patient_002.png
    └── ...
```

**Mask format**: Pixel values = class indices (0, 1, 2, 3, ...)

### Step 2: Define Your Classes
```python
# Example: CT-org dataset
CT_ORG_CLASSES = [
    "background",     # 0
    "liver",         # 1  
    "bladder",       # 2
    "lungs",         # 3
    "kidneys",       # 4
    "bone",          # 5
    # ... etc
]

# Example: Brain MRI
BRAIN_CLASSES = [
    "background",           # 0
    "cerebrospinal_fluid",  # 1
    "gray_matter",          # 2  
    "white_matter",         # 3
    "ventricles",           # 4
    # ... etc
]
```

### Step 3: Create Registers
```python
from anatomical_ldm import create_general_supervised_registers

# Automatically creates one register per class
registers = create_general_supervised_registers(
    num_classes=len(CT_ORG_CLASSES),
    class_names=CT_ORG_CLASSES,
    latent_resolution=8,
    d_model=512,
)

print(f"Created {len(CT_ORG_CLASSES)} registers:")
for i, name in enumerate(CT_ORG_CLASSES):
    print(f"  Register {i}: {name}")
```

### Step 4: Training
```python
# Progressive training strategy
trainer = setup_training(...)

# Stage 1: Strong supervision (learn class specialization)
trainer.anatomical_supervision_weight = 2.0
trainer.supervision_probability = 1.0  
trainer.train(num_epochs=100)

# Stage 2: Balanced supervision  
trainer.anatomical_supervision_weight = 0.5
trainer.supervision_probability = 0.6
trainer.train(num_epochs=200)

# Stage 3: Fine-tuning
trainer.anatomical_supervision_weight = 0.1
trainer.supervision_probability = 0.3
trainer.train(num_epochs=100)
```

### Step 5: Inference (No Masks Needed!)
```python
# At test time: uses learned anatomical knowledge
samples = pipeline(
    prompt="generate medical image",
    num_inference_steps=50,
    height=512, width=512
)
# Each register automatically provides appropriate anatomical conditioning
```

## 📊 What Each Register Learns

During training with masks, each register learns:

1. **Class Recognition**: "I am the liver register, I predict liver pixels"
2. **Spatial Priors**: "Liver typically appears in the upper-right region"  
3. **Anatomical Relationships**: "Liver is near gallbladder, below lungs"
4. **Stage Awareness**: "Emphasize layout early, details late in diffusion"

At test time, registers use this learned knowledge to provide anatomical conditioning.

## 🔍 Example Datasets Supported

### CT-org (Abdominal CT)
```python
CT_ORG_CLASSES = ["background", "liver", "kidneys", "spleen", "pancreas", ...]
registers = create_general_supervised_registers(num_classes=14, class_names=CT_ORG_CLASSES)
```

### Brain MRI  
```python
BRAIN_CLASSES = ["background", "gray_matter", "white_matter", "ventricles", ...]
registers = create_general_supervised_registers(num_classes=8, class_names=BRAIN_CLASSES)
```

### Cardiac CT
```python
CARDIAC_CLASSES = ["background", "myocardium", "left_ventricle", "right_ventricle", ...]
registers = create_general_supervised_registers(num_classes=10, class_names=CARDIAC_CLASSES)
```

### Chest X-ray (Still Works!)
```python
CHEST_CLASSES = ["background", "heart", "left_lung", "right_lung", "ribs", ...]
registers = create_general_supervised_registers(num_classes=12, class_names=CHEST_CLASSES)
```

## ⚙️ Training Parameters

### Required Parameters
- `num_classes`: Number of classes in your segmentation masks
- `class_names`: List of human-readable class names  

### Important Parameters
- `anatomical_supervision_weight`: Weight for anatomical loss (start high: 1.0-2.0)
- `supervision_probability`: Fraction of batches to use supervision (0.6-1.0)
- `enable_spatial_relationships`: Learn spatial relationships between classes (True)

### Progressive Training Strategy
```python
# Strong supervision → Balanced → Minimal supervision
weights = [2.0, 0.5, 0.1]  
probabilities = [1.0, 0.6, 0.3]
epochs = [100, 200, 100]
```

## 🎯 Expected Improvements

### With Supervised Registers:
✅ **Consistent anatomical positioning** (organs in right places)  
✅ **Realistic size ratios** (liver bigger than gallbladder)  
✅ **Proper relationships** (lungs bilateral, heart left-of-center)  
✅ **Controllable generation** (emphasize/de-emphasize organs)  

### Without Supervision (Baseline):
❌ Random organ positioning  
❌ Unrealistic anatomical relationships  
❌ Inconsistent generation quality  
❌ No anatomical controllability  

## 🚨 Common Issues & Solutions

### Issue: "Register collapse - all registers output same values"
**Solution**: Increase `anatomical_supervision_weight` and use progressive training

### Issue: "Poor spatial relationships"  
**Solution**: Enable `enable_spatial_relationships=True` and increase dataset size

### Issue: "Masks not loading correctly"
**Solution**: Check mask format (class indices: 0,1,2,...) and file extensions

### Issue: "OOM during training"
**Solution**: Reduce batch size, image resolution, or use gradient checkpointing

## 💡 Best Practices

1. **Start with strong supervision** (weight=2.0) to force register specialization
2. **Use progressive training** to gradually reduce supervision dependence  
3. **Provide meaningful class names** for interpretability
4. **Monitor attention maps** to ensure registers learn correct spatial priors
5. **Use sufficient data** (5k+ images minimum, 20k+ optimal)

## 🏆 Success Criteria

Your model is working if:
- ✅ Different registers generate different attention patterns
- ✅ Attention patterns align with anatomical structures  
- ✅ Generated images have consistent anatomical layout
- ✅ Anatomical accuracy improves vs baseline

---

**Ready to use with any multiclass medical imaging dataset!** 🚀