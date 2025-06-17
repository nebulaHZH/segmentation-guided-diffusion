#!/usr/bin/env python3
"""
Analyze the impact of anatomical registers on generation by manipulating gate values.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import diffusers
from diffusers import DDIMScheduler
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
from diagnose_registers import load_model_with_registers

class GateOverrideWrapper(nn.Module):
    """Wrapper to override gate values for analysis."""
    def __init__(self, model, gate_value=None):
        super().__init__()
        self.model = model
        self.override_gate = gate_value
        
    def forward(self, x, timestep, class_labels=None, return_dict=True):
        # Store original gate forward method
        original_gate_forward = self.model.gate.forward
        
        if self.override_gate is not None:
            # Override gate to return constant value
            def constant_gate(x):
                batch_size = x.shape[0]
                return torch.full((batch_size, 1), self.override_gate, device=x.device)
            self.model.gate.forward = constant_gate
        
        # Forward pass
        output = self.model(x, timestep, class_labels, return_dict)
        
        # Restore original gate
        self.model.gate.forward = original_gate_forward
        
        return output

def generate_with_gate_values(model, gate_values, num_samples=4, seed=42, device='cuda'):
    """Generate samples with different fixed gate values."""
    torch.manual_seed(seed)
    
    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(50)  # 50 denoising steps
    
    all_samples = {}
    
    for gate_value in gate_values:
        print(f"\nGenerating with gate value: {gate_value}")
        
        # Wrap model with gate override
        wrapped_model = GateOverrideWrapper(model, gate_value)
        samples = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Start from noise
                image = torch.randn(1, 1, 64, 64, device=device)
                
                # Denoising loop
                for t in tqdm(scheduler.timesteps, desc=f"Denoising (gate={gate_value})"):
                    timesteps = torch.full((1,), t, device=device, dtype=torch.long)
                    
                    # Predict noise
                    noise_pred = wrapped_model(image, timesteps, return_dict=True).sample
                    
                    # Denoise
                    image = scheduler.step(noise_pred, t, image).prev_sample
                
                # Convert to PIL image
                img_array = image.squeeze().cpu().numpy()
                img_array = (img_array + 1) / 2  # [-1, 1] to [0, 1]
                img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                samples.append(img_array)
        
        all_samples[gate_value] = samples
    
    return all_samples

def analyze_register_influence(model, device='cuda'):
    """Analyze how much registers influence the output at different timesteps."""
    print("\n=== Analyzing Register Influence ===")
    
    batch_size = 32
    timesteps_to_test = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    
    influences = []
    
    with torch.no_grad():
        for t in tqdm(timesteps_to_test, desc="Testing timesteps"):
            # Random input
            x = torch.randn(batch_size, 1, 64, 64).to(device)
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get output with normal gate
            output_normal = model(x, timesteps, return_dict=True).sample
            
            # Get output with gate=0 (no registers)
            wrapped_zero = GateOverrideWrapper(model, 0.0)
            output_zero = wrapped_zero(x, timesteps, return_dict=True).sample
            
            # Get output with gate=1 (full registers)
            wrapped_one = GateOverrideWrapper(model, 1.0)
            output_one = wrapped_one(x, timesteps, return_dict=True).sample
            
            # Compute differences
            diff_normal_zero = torch.mean(torch.abs(output_normal - output_zero)).item()
            diff_normal_one = torch.mean(torch.abs(output_normal - output_one)).item()
            diff_zero_one = torch.mean(torch.abs(output_zero - output_one)).item()
            
            influences.append({
                'timestep': t,
                'diff_normal_zero': diff_normal_zero,
                'diff_normal_one': diff_normal_one,
                'diff_zero_one': diff_zero_one
            })
    
    # Plot influence over timesteps
    plt.figure(figsize=(10, 6))
    
    timesteps_arr = [d['timestep'] for d in influences]
    diff_normal_zero = [d['diff_normal_zero'] for d in influences]
    diff_normal_one = [d['diff_normal_one'] for d in influences]
    diff_zero_one = [d['diff_zero_one'] for d in influences]
    
    plt.plot(timesteps_arr, diff_normal_zero, 'b-', label='Normal vs No Registers', marker='o')
    plt.plot(timesteps_arr, diff_normal_one, 'r-', label='Normal vs Full Registers', marker='s')
    plt.plot(timesteps_arr, diff_zero_one, 'g-', label='No Registers vs Full Registers', marker='^')
    
    plt.xlabel('Timestep')
    plt.ylabel('L1 Difference')
    plt.title('Register Influence Across Denoising Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('register_influence.png', dpi=150)
    plt.close()
    
    return influences

def visualize_gate_spectrum(samples_dict):
    """Create a grid showing samples with different gate values."""
    gate_values = sorted(samples_dict.keys())
    num_samples = len(list(samples_dict.values())[0])
    
    fig, axes = plt.subplots(len(gate_values), num_samples, 
                            figsize=(num_samples * 2, len(gate_values) * 2))
    
    if len(gate_values) == 1:
        axes = axes.reshape(1, -1)
    
    for i, gate_value in enumerate(gate_values):
        for j, img in enumerate(samples_dict[gate_value]):
            ax = axes[i, j] if len(gate_values) > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Gate = {gate_value}', fontsize=12, pad=10)
    
    plt.suptitle('Generated Samples with Different Gate Values', fontsize=16)
    plt.tight_layout()
    plt.savefig('gate_spectrum_samples.png', dpi=200, bbox_inches='tight')
    plt.close()

def check_register_weight_magnitudes(model):
    """Check if register weights have collapsed or grown too large."""
    print("\n=== Checking Register Weight Health ===")
    
    # Check projection layer
    proj_weight = model.register_proj.weight
    proj_mag = torch.norm(proj_weight).item()
    print(f"Projection layer weight magnitude: {proj_mag:.4f}")
    
    # Check gate network
    gate_weights = []
    for i, layer in enumerate(model.gate):
        if hasattr(layer, 'weight'):
            weight_mag = torch.norm(layer.weight).item()
            gate_weights.append(weight_mag)
            print(f"Gate layer {i} weight magnitude: {weight_mag:.4f}")
    
    # Check if weights are too small (collapsed) or too large
    if proj_mag < 0.01:
        print("WARNING: Projection weights may have collapsed!")
    if any(w < 0.01 for w in gate_weights):
        print("WARNING: Some gate weights may have collapsed!")
        
    # Check output range of gate
    test_input = torch.randn(10, model.register_bank.dim).to(next(model.parameters()).device)
    with torch.no_grad():
        gate_outputs = model.gate(test_input)
    print(f"\nGate output range on random input: [{gate_outputs.min().item():.4f}, {gate_outputs.max().item():.4f}]")

def analyze_register_gradients_detailed(model, img_size=64, device='cuda'):
    """Detailed analysis of gradient flow through registers."""
    print("\n=== Detailed Gradient Analysis ===")
    
    model.train()
    
    # Test forward and backward pass
    x = torch.randn(4, 1, img_size, img_size, device=device, requires_grad=True)
    timesteps = torch.randint(0, 1000, (4,), device=device)
    
    # Forward pass with gradient tracking
    output = model(x, timesteps, return_dict=True).sample
    
    # Create a simple loss
    loss = output.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nGradient presence check:")
    
    # Registers
    if model.register_bank.organ_registers.grad is not None:
        print(f"✓ Organ registers have gradients: {model.register_bank.organ_registers.grad.norm().item():.6f}")
    else:
        print("✗ Organ registers have NO gradients!")
        
    if model.register_bank.spatial_registers.grad is not None:
        print(f"✓ Spatial registers have gradients: {model.register_bank.spatial_registers.grad.norm().item():.6f}")
    else:
        print("✗ Spatial registers have NO gradients!")
        
    if model.register_bank.scale_registers.grad is not None:
        print(f"✓ Scale registers have gradients: {model.register_bank.scale_registers.grad.norm().item():.6f}")
    else:
        print("✗ Scale registers have NO gradients!")
    
    # Projection and gate
    if model.register_proj.weight.grad is not None:
        print(f"✓ Projection layer has gradients: {model.register_proj.weight.grad.norm().item():.6f}")
    else:
        print("✗ Projection layer has NO gradients!")
        
    # Gate layers
    for i, layer in enumerate(model.gate):
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            print(f"✓ Gate layer {i} has gradients: {layer.weight.grad.norm().item():.6f}")
        elif hasattr(layer, 'weight'):
            print(f"✗ Gate layer {i} has NO gradients!")
    
    model.eval()
    model.zero_grad()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anatomical_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=4)
    
    args = parser.parse_args()
    
    # Load model
    print("Loading anatomical model...")
    model = load_model_with_registers(args.anatomical_dir, args.device)
    
    # 1. Check weight health
    check_register_weight_magnitudes(model)
    
    # 2. Analyze gradient flow
    analyze_register_gradients_detailed(model, device=args.device)
    
    # 3. Generate samples with different gate values
    gate_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    samples = generate_with_gate_values(model, gate_values, 
                                      num_samples=args.num_samples, 
                                      device=args.device)
    visualize_gate_spectrum(samples)
    
    # 4. Analyze register influence
    influences = analyze_register_influence(model, device=args.device)
    
    print("\n=== Analysis Complete ===")
    print("Generated visualizations:")
    print("  - gate_spectrum_samples.png: Samples with different gate values")
    print("  - register_influence.png: Register influence across timesteps")

if __name__ == "__main__":
    main()