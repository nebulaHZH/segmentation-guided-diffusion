# Anatomical Registers Diagnosis Guide

## Quick Diagnosis Steps

### 1. First, run the quick gate check:
```bash
python quick_gate_check.py /path/to/ddim-chexpert_subset-64-anatomical
```

This will immediately tell you if the registers are being used at all. If you see:
- **Gate mean < 0.1**: The model has learned to ignore registers (most likely issue)
- **Gate mean ≈ 0.5**: Registers are being used moderately
- **Gate mean > 0.8**: Model is over-relying on registers

### 2. Run comprehensive diagnosis:
```bash
python diagnose_registers.py \
    --anatomical_dir /path/to/ddim-chexpert_subset-64-anatomical \
    --baseline_dir /path/to/ddim-chexpert_subset-64 \
    --num_samples 1000
```

This generates:
- `gate_analysis.png`: Shows gate values across timesteps
- `register_similarities.png`: Shows if registers learned distinct patterns
- `stage_analysis.png`: Shows stage-specific register usage
- `baseline_comparison.png`: Compares outputs with baseline

### 3. Test register impact:
```bash
python analyze_register_impact.py \
    --anatomical_dir /path/to/ddim-chexpert_subset-64-anatomical \
    --num_samples 4
```

This generates:
- `gate_spectrum_samples.png`: Shows samples with gate forced to 0, 0.25, 0.5, 0.75, 1.0
- `register_influence.png`: Shows register influence across denoising steps

## Common Issues and Solutions

### Issue 1: Gate Values Near Zero (Most Likely)
**Symptoms**: 
- Gate mean < 0.1
- No visual difference between baseline and anatomical models
- Same FID/FRD scores

**Why it happens**:
- Registers start random, initially hurt performance
- With large batch sizes, model quickly learns to ignore them
- Current architecture only modulates final output (too late in pipeline)

**Solutions**:
1. **Better initialization**: Initialize registers from pretrained features
2. **Gradual introduction**: Start with gate=0, gradually increase during training
3. **Deeper integration**: Modulate intermediate UNet features, not just output
4. **Auxiliary losses**: Add losses that encourage register usage

### Issue 2: Registers Not Learning Distinct Patterns
**Symptoms**:
- High similarity (>0.8) between registers of same type
- Registers converge to similar patterns

**Solutions**:
1. **Diversity loss**: Penalize high similarity between registers
2. **Specialized initialization**: Initialize each register differently
3. **Register-specific losses**: Different objectives for different registers

### Issue 3: Poor Stage-Aware Behavior
**Symptoms**:
- Similar register weights across all timesteps
- No clear layout vs detail distinction

**Solutions**:
1. **Stronger stage conditioning**: Make stage differences more pronounced
2. **Stage-specific architectures**: Different processing for different stages

## Recommended Fixes

### Fix 1: Warmup Strategy for Gate
Add this to training:

```python
class WarmupGate(nn.Module):
    def __init__(self, gate_module, warmup_epochs=50, max_value=0.5):
        super().__init__()
        self.gate = gate_module
        self.warmup_epochs = warmup_epochs
        self.max_value = max_value
        self.current_epoch = 0
        
    def forward(self, x):
        base_gate = self.gate(x)
        if self.training:
            # Gradually increase gate influence
            warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
            return base_gate * warmup_factor * self.max_value
        return base_gate
```

### Fix 2: Intermediate Feature Modulation
Instead of modulating only the output, modulate intermediate features:

```python
class ImprovedRegisterModulation(nn.Module):
    def forward(self, x, timestep, ...):
        # Get registers
        registers = self.register_bank(x, timestep)
        
        # Hook into UNet intermediate layers
        def modulate_hook(module, input, output):
            # Modulate intermediate features
            gate = self.compute_gate(registers, output)
            return output * (1 - gate) + register_features * gate
            
        # Register hooks on UNet blocks
        hooks = []
        for block in self.unet.down_blocks + self.unet.up_blocks:
            hook = block.register_forward_hook(modulate_hook)
            hooks.append(hook)
        
        # Forward pass
        output = self.unet(x, timestep, ...)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return output
```

### Fix 3: Auxiliary Register Loss
Add a loss that encourages register usage:

```python
def register_loss(model, x, timestep, alpha=0.1):
    # Get outputs with and without registers
    with torch.no_grad():
        model.gate.eval()  # Temporarily set gate to eval mode
        old_forward = model.gate.forward
        model.gate.forward = lambda x: torch.zeros_like(x)[:, :1]
        output_no_reg = model(x, timestep)
        model.gate.forward = old_forward
        model.gate.train()
    
    output_with_reg = model(x, timestep)
    
    # Encourage difference when registers are used
    reg_diff = torch.mean(torch.abs(output_with_reg - output_no_reg))
    
    # Also encourage meaningful gate values
    gate_value = model.gate(model.register_bank(x, timestep)["registers"].mean(1))
    gate_loss = -torch.log(gate_value + 1e-8).mean()  # Encourage non-zero gates
    
    return alpha * (gate_loss - reg_diff)
```

## Quick Experiments to Try

1. **Force gate to 0.3 and retrain briefly**:
   ```python
   # In forward method, replace gate computation with:
   gate_value = torch.full_like(gate_value, 0.3)
   ```
   This tests if registers can learn useful patterns when forced to be used.

2. **Initialize registers from image statistics**:
   ```python
   # Instead of random init, use PCA of training images
   with torch.no_grad():
       image_features = extract_features_from_training_set()
       pca_components = compute_pca(image_features, n_components=22)
       model.register_bank.organ_registers.data = pca_components[:12]
       # etc.
   ```

3. **Add register regularization to training loss**:
   ```python
   total_loss = mse_loss + 0.01 * register_loss(model, noisy_images, timesteps)
   ```

The most likely issue is that the gate learned to output near-zero values, effectively disabling the anatomical registers. Run the diagnostics to confirm, then implement the fixes above.