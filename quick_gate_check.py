#!/usr/bin/env python3
"""
Quick diagnostic to check what gate values the model has learned.
This is the most critical diagnostic - if gate ≈ 0, registers are being ignored.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

def quick_gate_check(checkpoint_dir, num_samples=100):
    """Quick check of gate values across different timesteps."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {checkpoint_dir}...")
    
    # Load base UNet
    unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(checkpoint_dir, 'unet'),
        use_safetensors=True
    )
    
    # Create and load anatomical registers
    register_bank = AnatomicalRegisterBank()
    model = RegisterModulatedUNet(unet, register_bank)
    
    register_path = Path(checkpoint_dir) / 'anatomical_registers.pt'
    if register_path.exists():
        state = torch.load(register_path, map_location=device)
        model.register_bank.load_state_dict(state['register_bank'])
        model.register_proj.load_state_dict(state['register_proj'])
        model.gate.load_state_dict(state['gate'])
        print("✓ Loaded anatomical registers")
    else:
        print("✗ No anatomical_registers.pt found!")
        return
    
    model.to(device).eval()
    
    # Test gate values at different timesteps
    print("\n" + "="*50)
    print("GATE VALUE ANALYSIS")
    print("="*50)
    
    timestep_ranges = [
        (0, 100, "Final details"),
        (100, 300, "Fine features"),
        (300, 500, "Mid denoising"),
        (500, 700, "Coarse features"),
        (700, 900, "Layout/structure"),
        (900, 1000, "Initial noise")
    ]
    
    all_gates = []
    
    with torch.no_grad():
        for t_min, t_max, stage_name in timestep_ranges:
            gates = []
            
            for _ in range(num_samples // len(timestep_ranges)):
                # Random input and timestep in range
                x = torch.randn(1, 1, 64, 64).to(device)
                t = torch.randint(t_min, t_max, (1,)).to(device)
                
                # Get registers and compute gate
                register_dict = model.register_bank(x, t)
                pooled = register_dict["registers"].mean(dim=1)
                gate = model.gate(pooled)
                gates.append(gate.item())
            
            gates = np.array(gates)
            all_gates.extend(gates)
            
            print(f"\n{stage_name} (t={t_min}-{t_max}):")
            print(f"  Mean gate: {np.mean(gates):.4f}")
            print(f"  Std gate:  {np.std(gates):.4f}")
            print(f"  Min gate:  {np.min(gates):.4f}")
            print(f"  Max gate:  {np.max(gates):.4f}")
    
    # Overall statistics
    all_gates = np.array(all_gates)
    print(f"\nOVERALL STATISTICS:")
    print(f"  Mean: {np.mean(all_gates):.4f}")
    print(f"  Std:  {np.std(all_gates):.4f}")
    print(f"  Min:  {np.min(all_gates):.4f}")
    print(f"  Max:  {np.max(all_gates):.4f}")
    
    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION:")
    print("="*50)
    
    mean_gate = np.mean(all_gates)
    if mean_gate < 0.1:
        print("⚠️  PROBLEM: Gate values are very low (mean < 0.1)")
        print("   The model is barely using the anatomical registers!")
        print("   Registers contribute only ~{:.1f}% to the output.".format(mean_gate * 100))
    elif mean_gate < 0.3:
        print("⚠️  WARNING: Gate values are low (mean < 0.3)")
        print("   Registers have limited influence (~{:.1f}% contribution).".format(mean_gate * 100))
    elif mean_gate > 0.7:
        print("⚠️  WARNING: Gate values are high (mean > 0.7)")
        print("   Model may be over-relying on registers.")
    else:
        print("✓  Gate values look reasonable (mean ≈ {:.2f})".format(mean_gate))
        print("   Registers contribute ~{:.1f}% to the output.".format(mean_gate * 100))
    
    # Check variation
    if np.std(all_gates) < 0.05:
        print("\n⚠️  Low variation in gate values (std < 0.05)")
        print("   Gate might be stuck at a constant value.")
    
    # Test actual modulation effect
    print("\n" + "="*50)
    print("MODULATION EFFECT TEST")
    print("="*50)
    
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        # Normal output
        output_normal = model(x, t, return_dict=True).sample
        
        # Get gate value
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        gate_value = model.gate(pooled).item()
        
        # Manually compute what pure UNet would output
        unet_output = model.unet(x, t, return_dict=True).sample
        
        # Check difference
        diff = torch.mean(torch.abs(output_normal - unet_output)).item()
        
        print(f"Difference between modulated and pure UNet output: {diff:.6f}")
        print(f"Gate value for this sample: {gate_value:.4f}")
        print(f"Expected difference magnitude: ~{gate_value * 0.5:.4f} (assuming register features differ from UNet)")
        
        if diff < 0.001:
            print("\n⚠️  PROBLEM: Almost no difference between modulated and pure UNet!")
            print("   Registers are having negligible effect.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_gate_check.py <checkpoint_dir>")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    quick_gate_check(checkpoint_dir)