#!/usr/bin/env python3
"""
Analyze if registers actually learned meaningful anatomical patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from pathlib import Path
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

def analyze_register_quality(checkpoint_dir, device='cuda'):
    """Check if registers learned meaningful patterns."""
    
    # Load model
    unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(checkpoint_dir, 'unet'), use_safetensors=True
    )
    register_bank = AnatomicalRegisterBank()
    model = RegisterModulatedUNet(unet, register_bank)
    
    register_path = Path(checkpoint_dir) / 'anatomical_registers.pt'
    if register_path.exists():
        state = torch.load(register_path, map_location=device)
        model.register_bank.load_state_dict(state['register_bank'])
        model.register_proj.load_state_dict(state['register_proj'])
        model.gate.load_state_dict(state['gate'])
    
    model.to(device).eval()
    
    print("="*60)
    print("REGISTER QUALITY ANALYSIS")
    print("="*60)
    
    # 1. Check if registers are diverse
    with torch.no_grad():
        organ_regs = model.register_bank.organ_registers.cpu().numpy()
        spatial_regs = model.register_bank.spatial_registers.cpu().numpy()
        scale_regs = model.register_bank.scale_registers.cpu().numpy()
        
        print("\n1. REGISTER DIVERSITY:")
        
        def analyze_diversity(regs, name):
            # Compute pairwise similarities
            norms = np.linalg.norm(regs, axis=1, keepdims=True)
            normalized = regs / (norms + 1e-8)
            similarities = np.dot(normalized, normalized.T)
            
            # Get off-diagonal elements
            n = similarities.shape[0]
            mask = ~np.eye(n, dtype=bool)
            off_diag = similarities[mask]
            
            print(f"\n{name} registers:")
            print(f"  Average similarity: {np.mean(off_diag):.4f}")
            print(f"  Similarity std: {np.std(off_diag):.4f}")
            print(f"  Min similarity: {np.min(off_diag):.4f}")
            print(f"  Max similarity: {np.max(off_diag):.4f}")
            
            if np.mean(off_diag) > 0.9:
                print(f"  ⚠️ HIGH SIMILARITY - registers are nearly identical!")
            elif np.mean(off_diag) > 0.7:
                print(f"  ⚠️ Moderate similarity - limited diversity")
            else:
                print(f"  ✓ Good diversity")
            
            return similarities
        
        organ_sim = analyze_diversity(organ_regs, "Organ")
        spatial_sim = analyze_diversity(spatial_regs, "Spatial") 
        scale_sim = analyze_diversity(scale_regs, "Scale")
    
    # 2. Check if registers respond to different inputs
    print("\n2. REGISTER RESPONSIVENESS:")
    
    with torch.no_grad():
        # Test on different types of inputs
        test_cases = [
            ("Noise", torch.randn(5, 1, 64, 64)),
            ("Zeros", torch.zeros(5, 1, 64, 64)),
            ("Ones", torch.ones(5, 1, 64, 64)),
            ("Checkerboard", torch.zeros(5, 1, 64, 64)),
        ]
        
        # Create checkerboard pattern
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                if (i//8 + j//8) % 2 == 0:
                    test_cases[3][1][:, :, i:i+8, j:j+8] = 1
        
        register_responses = {}
        
        for name, inputs in test_cases:
            inputs = inputs.to(device)
            timesteps = torch.tensor([500] * inputs.shape[0]).to(device)
            
            register_dict = model.register_bank(inputs, timesteps)
            pooled = register_dict["registers"].mean(dim=1)
            
            register_responses[name] = pooled.cpu().numpy()
            
            print(f"\n{name} input:")
            print(f"  Pooled register mean: {np.mean(register_responses[name]):.6f}")
            print(f"  Pooled register std: {np.std(register_responses[name]):.6f}")
        
        # Check if responses are different
        print(f"\nRegister response differences:")
        baseline = register_responses["Noise"]
        for name, response in register_responses.items():
            if name != "Noise":
                diff = np.mean(np.abs(response - baseline))
                print(f"  {name} vs Noise: {diff:.6f}")
                if diff < 0.01:
                    print(f"    ⚠️ Almost identical responses!")
    
    # 3. Check register projection output scale
    print("\n3. REGISTER PROJECTION ANALYSIS:")
    
    with torch.no_grad():
        # Get projection weights and biases
        proj_weight = model.register_proj.weight.cpu().numpy()
        proj_bias = model.register_proj.bias.cpu().numpy() if model.register_proj.bias is not None else None
        
        print(f"Projection weight shape: {proj_weight.shape}")
        print(f"Projection weight magnitude: {np.linalg.norm(proj_weight):.6f}")
        print(f"Projection weight mean: {np.mean(proj_weight):.6f}")
        print(f"Projection weight std: {np.std(proj_weight):.6f}")
        
        if proj_bias is not None:
            print(f"Projection bias: {proj_bias}")
        
        # Test projection output scale vs UNet output scale
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        unet_out = model.unet(x, t, return_dict=True).sample
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        projected = model.register_proj(pooled)
        
        print(f"\nOutput scales:")
        print(f"  UNet output: mean={unet_out.mean().item():.4f}, std={unet_out.std().item():.4f}")
        print(f"  Register projection: {projected.cpu().numpy().flatten()}")
        
        scale_ratio = torch.std(unet_out) / (torch.abs(projected).mean() + 1e-8)
        print(f"  Scale ratio (UNet/Register): {scale_ratio.item():.2f}")
        
        if scale_ratio > 100:
            print("  ⚠️ Register features are MUCH smaller than UNet output!")
        elif scale_ratio < 0.01:
            print("  ⚠️ Register features are MUCH larger than UNet output!")
    
    # 4. Test actual impact on generation
    print("\n4. GENERATION IMPACT TEST:")
    
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        # Get normal output
        normal_out = model(x, t, return_dict=True).sample
        
        # Get pure UNet output (what would happen with gate=0)
        unet_only = model.unet(x, t, return_dict=True).sample
        
        # Compute actual difference
        actual_diff = torch.mean(torch.abs(normal_out - unet_only)).item()
        
        # Get the gate value used
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        gate_val = model.gate(pooled).item()
        
        print(f"  Gate value: {gate_val:.4f}")
        print(f"  Actual output difference: {actual_diff:.6f}")
        print(f"  Expected difference if meaningful: ~{gate_val * unet_out.std().item():.6f}")
        
        efficiency = actual_diff / (gate_val * unet_out.std().item() + 1e-8)
        print(f"  Modulation efficiency: {efficiency:.4f}")
        
        if efficiency < 0.1:
            print("  ⚠️ Registers have almost no effect despite gate value!")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    # Determine if registers are actually useful
    issues = []
    
    if np.mean([np.mean(organ_sim > 0.9), np.mean(spatial_sim > 0.9), np.mean(scale_sim > 0.9)]) > 0.5:
        issues.append("Registers are too similar (collapsed)")
    
    # Check if all register responses are similar
    if all(np.mean(np.abs(resp - baseline)) < 0.01 for name, resp in register_responses.items() if name != "Noise"):
        issues.append("Registers don't respond differently to different inputs")
    
    if scale_ratio > 50 or scale_ratio < 0.02:
        issues.append("Register features are wrong scale vs UNet output")
    
    if efficiency < 0.1:
        issues.append("Registers have negligible impact on output")
    
    if issues:
        print("❌ REGISTERS ARE NOT FUNCTIONAL:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThe gate collapse was CORRECT - registers aren't helpful!")
        print("Fixing the gate will likely NOT improve performance.")
    else:
        print("✅ Registers appear functional but underutilized")
        print("Gate fix might help if registers can learn better patterns.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_register_quality.py <checkpoint_dir>")
        sys.exit(1)
    
    analyze_register_quality(sys.argv[1])