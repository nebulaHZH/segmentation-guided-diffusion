#!/usr/bin/env python3
"""
Diagnose why the gate network collapsed to a constant value.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

def diagnose_gate_collapse(checkpoint_dir, device='cuda'):
    """Comprehensive diagnosis of gate collapse issue."""
    
    print("Loading model...")
    # Load model components
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
    
    print("\n" + "="*60)
    print("GATE NETWORK WEIGHT ANALYSIS")
    print("="*60)
    
    # Analyze gate network weights
    for i, layer in enumerate(model.gate):
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            bias = layer.bias.data if hasattr(layer, 'bias') else None
            
            print(f"\nLayer {i} ({layer.__class__.__name__}):")
            print(f"  Weight shape: {weight.shape}")
            print(f"  Weight magnitude: {torch.norm(weight).item():.6f}")
            print(f"  Weight mean: {weight.mean().item():.6f}")
            print(f"  Weight std: {weight.std().item():.6f}")
            
            if bias is not None:
                print(f"  Bias: {bias.data.cpu().numpy()}")
            
            # Check if weights are near zero (dead neurons)
            if torch.norm(weight).item() < 0.01:
                print("  ⚠️  WARNING: Very small weights - possible dead neurons!")
    
    print("\n" + "="*60)
    print("REGISTER DIVERSITY ANALYSIS")
    print("="*60)
    
    # Check if registers are diverse or collapsed
    with torch.no_grad():
        # Test on different inputs
        test_inputs = []
        test_timesteps = []
        
        for _ in range(10):
            x = torch.randn(1, 1, 64, 64).to(device)
            t = torch.randint(0, 1000, (1,)).to(device)
            test_inputs.append(x)
            test_timesteps.append(t)
        
        # Get register outputs for different inputs
        all_pooled_registers = []
        for x, t in zip(test_inputs, test_timesteps):
            register_dict = model.register_bank(x, t)
            pooled = register_dict["registers"].mean(dim=1)
            all_pooled_registers.append(pooled)
        
        # Stack and analyze
        all_pooled = torch.cat(all_pooled_registers, dim=0)
        register_mean = all_pooled.mean(dim=0)
        register_std = all_pooled.std(dim=0)
        
        print(f"Pooled register statistics across different inputs:")
        print(f"  Mean magnitude: {torch.norm(register_mean).item():.6f}")
        print(f"  Std magnitude: {torch.norm(register_std).item():.6f}")
        print(f"  Min std: {register_std.min().item():.6f}")
        print(f"  Max std: {register_std.max().item():.6f}")
        
        if torch.norm(register_std).item() < 0.01:
            print("  ⚠️  WARNING: Registers have very low variance across inputs!")
    
    print("\n" + "="*60)
    print("GRADIENT FLOW TEST")
    print("="*60)
    
    # Test if gradients flow to gate
    model.train()
    model.zero_grad()
    
    x = torch.randn(1, 1, 64, 64, device=device, requires_grad=True)
    t = torch.tensor([500], device=device)
    
    # Forward pass
    output = model(x, t, return_dict=True).sample
    
    # Create a loss that should depend on gate
    # We'll create a target that's different from both pure UNet and pure registers
    with torch.no_grad():
        # Get pure UNet output
        unet_output = model.unet(x, t, return_dict=True).sample
        # Get register features
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        register_features = model.register_proj(pooled)
        register_features = register_features.view(1, -1, 1, 1).expand_like(unet_output)
        
        # Create target that's different from both
        target = unet_output * 0.3 + register_features * 0.7 + torch.randn_like(unet_output) * 0.1
    
    # Loss that should encourage gate ≈ 0.7
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    
    print("Gradient magnitudes after backward pass:")
    
    # Check gate gradients
    gate_has_grad = False
    for i, layer in enumerate(model.gate):
        if hasattr(layer, 'weight') and layer.weight.grad is not None:
            grad_norm = torch.norm(layer.weight.grad).item()
            print(f"  Gate layer {i}: {grad_norm:.8f}")
            if grad_norm > 1e-8:
                gate_has_grad = True
        elif hasattr(layer, 'weight'):
            print(f"  Gate layer {i}: NO GRADIENT")
    
    if not gate_has_grad:
        print("  ⚠️  CRITICAL: Gate network receives no gradients!")
    
    # Check register gradients
    if model.register_bank.organ_registers.grad is not None:
        print(f"  Organ registers: {torch.norm(model.register_bank.organ_registers.grad).item():.8f}")
    else:
        print("  Organ registers: NO GRADIENT")
    
    print("\n" + "="*60)
    print("FORWARD PASS BREAKDOWN")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Trace through a forward pass
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        # Get registers
        register_dict = model.register_bank(x, t)
        registers = register_dict["registers"]
        print(f"Registers shape: {registers.shape}")
        print(f"Registers mean: {registers.mean().item():.6f}")
        print(f"Registers std: {registers.std().item():.6f}")
        
        # Pool registers
        pooled = registers.mean(dim=1)
        print(f"\nPooled registers shape: {pooled.shape}")
        print(f"Pooled registers mean: {pooled.mean().item():.6f}")
        print(f"Pooled registers std: {pooled.std().item():.6f}")
        
        # Gate computation
        gate_input = pooled
        
        # Manual forward through gate
        for i, layer in enumerate(model.gate):
            if isinstance(layer, nn.Linear):
                gate_input = layer(gate_input)
                print(f"\nAfter gate layer {i}:")
                print(f"  Output: {gate_input.cpu().numpy().flatten()[:5]}...")
                print(f"  Mean: {gate_input.mean().item():.6f}")
                print(f"  Std: {gate_input.std().item():.6f}")
            elif isinstance(layer, nn.GELU):
                gate_input = layer(gate_input)
                print(f"\nAfter GELU:")
                print(f"  Output: {gate_input.cpu().numpy().flatten()[:5]}...")
            elif isinstance(layer, nn.Sigmoid):
                print(f"\nBefore sigmoid: {gate_input.item():.6f}")
                gate_input = layer(gate_input)
                print(f"After sigmoid: {gate_input.item():.6f}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    # The gate outputs exactly 0.455, which means pre-sigmoid value is:
    # sigmoid(x) = 0.455, so x = log(0.455 / (1 - 0.455)) ≈ -0.180
    expected_pre_sigmoid = np.log(0.455 / (1 - 0.455))
    print(f"Expected pre-sigmoid value for gate=0.455: {expected_pre_sigmoid:.6f}")
    print("\nPossible causes:")
    print("1. Final linear layer in gate has learned to output constant -0.18")
    print("2. All inputs to gate network are identical (collapsed registers)")
    print("3. Gate network weights have been zeroed out or made constant")
    print("4. Gradient flow to gate is blocked or too weak")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python diagnose_gate_collapse.py <checkpoint_dir>")
        sys.exit(1)
    
    diagnose_gate_collapse(sys.argv[1])