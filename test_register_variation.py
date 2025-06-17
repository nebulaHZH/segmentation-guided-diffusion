#!/usr/bin/env python3
"""
Test if registers and gate can produce different outputs when forced.
"""

import torch
import numpy as np
import os
from pathlib import Path
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers
from tqdm import tqdm

def test_register_variation(checkpoint_dir, device='cuda'):
    # Load model
    print("Loading model...")
    unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(checkpoint_dir, 'unet'),
        use_safetensors=True
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
    
    print("\n=== Testing Register Outputs ===")
    
    # Test 1: Do registers produce different outputs for different timesteps?
    with torch.no_grad():
        timesteps = [100, 500, 900]
        register_outputs = []
        
        for t in timesteps:
            x = torch.randn(1, 1, 64, 64).to(device)
            t_tensor = torch.tensor([t]).to(device)
            
            register_dict = model.register_bank(x, t_tensor)
            pooled = register_dict["registers"].mean(dim=1)
            register_outputs.append(pooled.cpu().numpy())
        
        # Check variation
        print("\nRegister outputs for different timesteps:")
        for i, t in enumerate(timesteps):
            print(f"  t={t}: mean={np.mean(register_outputs[i]):.4f}, std={np.std(register_outputs[i]):.4f}")
        
        # Compare similarity between timesteps
        for i in range(len(timesteps)):
            for j in range(i+1, len(timesteps)):
                diff = np.mean(np.abs(register_outputs[i] - register_outputs[j]))
                print(f"  Difference t={timesteps[i]} vs t={timesteps[j]}: {diff:.6f}")
    
    print("\n=== Testing Gate Input/Output Mapping ===")
    
    # Test 2: What happens with very different inputs to gate?
    with torch.no_grad():
        # Create artificial inputs spanning a range
        test_inputs = [
            torch.zeros(1, 512).to(device),
            torch.ones(1, 512).to(device) * 0.1,
            torch.ones(1, 512).to(device),
            torch.randn(1, 512).to(device) * 0.01,
            torch.randn(1, 512).to(device),
            torch.randn(1, 512).to(device) * 10,
        ]
        
        print("\nGate outputs for different artificial inputs:")
        for i, inp in enumerate(test_inputs):
            gate_out = model.gate(inp)
            print(f"  Input {i} (norm={torch.norm(inp).item():.2f}): gate={gate_out.item():.6f}")
    
    print("\n=== Testing Modified Gate ===")
    
    # Test 3: What if we force the gate to output different values?
    old_gate_forward = model.gate.forward
    
    def test_with_fixed_gate(gate_value):
        # Override gate
        model.gate.forward = lambda x: torch.full((x.shape[0], 1), gate_value, device=x.device)
        
        # Test output
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        with torch.no_grad():
            output = model(x, t, return_dict=True).sample
            
        # Restore
        model.gate.forward = old_gate_forward
        
        return output
    
    print("\nOutput statistics with forced gate values:")
    baseline_output = None
    for gate_val in [0.0, 0.2, 0.455, 0.7, 1.0]:
        output = test_with_fixed_gate(gate_val)
        print(f"  Gate={gate_val}: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        
        if baseline_output is None:
            baseline_output = output
        else:
            diff = torch.mean(torch.abs(output - baseline_output)).item()
            print(f"    Difference from gate=0.0: {diff:.6f}")
    
    print("\n=== Testing Register Projection ===")
    
    # Test 4: Check if register projection produces meaningful features
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        # Get registers and project
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        projected = model.register_proj(pooled)
        
        print(f"\nRegister projection output:")
        print(f"  Shape: {projected.shape}")
        print(f"  Mean: {projected.mean().item():.6f}")
        print(f"  Std: {projected.std().item():.6f}")
        print(f"  Min: {projected.min().item():.6f}")
        print(f"  Max: {projected.max().item():.6f}")
        
        # Compare with UNet output scale
        unet_output = model.unet(x, t, return_dict=True).sample
        print(f"\nUNet output scale:")
        print(f"  Mean: {unet_output.mean().item():.6f}")
        print(f"  Std: {unet_output.std().item():.6f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_register_variation.py <checkpoint_dir>")
        sys.exit(1)
    
    test_register_variation(sys.argv[1])