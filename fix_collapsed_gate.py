#!/usr/bin/env python3
"""
Fix for collapsed gate by adding noise and retraining.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

class FixedGateWrapper(nn.Module):
    """Wrapper that fixes gate behavior during training."""
    
    def __init__(self, original_gate, noise_std=0.1, min_val=0.1, max_val=0.7):
        super().__init__()
        self.original_gate = original_gate
        self.noise_std = noise_std
        self.min_val = min_val
        self.max_val = max_val
        
    def forward(self, x):
        # Get original gate output
        base_output = self.original_gate(x)
        
        if self.training:
            # Add noise to break constant behavior
            noise = torch.randn_like(base_output) * self.noise_std
            output = base_output + noise
            
            # Clamp to reasonable range
            output = torch.clamp(output, self.min_val, self.max_val)
        else:
            output = base_output
            
        return output

def apply_gate_fix(model):
    """Apply fix to a collapsed gate."""
    print("Applying gate fix...")
    
    # Wrap the gate with noise injection
    model.gate = FixedGateWrapper(model.gate, noise_std=0.1, min_val=0.1, max_val=0.7)
    
    # Reinitialize gate network weights with small values
    for layer in model.gate.original_gate:
        if isinstance(layer, nn.Linear):
            # Small random weights
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    print("Gate fix applied. Model now has:")
    print("- Noise injection during training (std=0.1)")
    print("- Gate values clamped to [0.1, 0.7]")
    print("- Reinitialized gate weights")

def load_and_fix_model(checkpoint_dir, device='cuda'):
    """Load model and apply gate fix."""
    # Load model
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
    
    # Apply fix
    apply_gate_fix(model)
    
    model.to(device)
    return model

# Alternative: Just replace the gate entirely
class VariableGate(nn.Module):
    """New gate that's designed to be more variable."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),  # Add normalization
            nn.GELU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize with very small weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # Add small random component to input to ensure variation
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        gate_val = self.net(x)
        
        # Ensure minimum and maximum values
        gate_val = 0.1 + 0.6 * gate_val  # Maps [0,1] to [0.1, 0.7]
        
        return gate_val

def replace_gate_entirely(model):
    """Replace the collapsed gate with a new one."""
    print("Replacing gate entirely...")
    
    # Get input dimension
    input_dim = model.register_bank.dim
    
    # Create new gate
    new_gate = VariableGate(input_dim)
    
    # Replace
    model.gate = new_gate
    
    print("Gate replaced with new architecture including:")
    print("- Layer normalization")
    print("- Dropout regularization") 
    print("- Input noise injection")
    print("- Forced range [0.1, 0.7]")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_collapsed_gate.py <checkpoint_dir> [--test-only]")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    test_only = "--test-only" in sys.argv
    
    if test_only:
        # Just test the current gate
        model = load_and_fix_model(checkpoint_dir)
        
        print("\nTesting fixed gate...")
        with torch.no_grad():
            for i in range(10):
                x = torch.randn(1, 1, 64, 64).cuda()
                t = torch.randint(0, 1000, (1,)).cuda()
                
                register_dict = model.register_bank(x, t)
                pooled = register_dict["registers"].mean(dim=1)
                gate_val = model.gate(pooled)
                
                print(f"  Sample {i}: gate = {gate_val.item():.4f}")
    else:
        print("Use this in your training script to fix the collapsed gate:")
        print("\n# Option 1: Wrap existing gate")
        print("apply_gate_fix(model)")
        print("\n# Option 2: Replace gate entirely") 
        print("replace_gate_entirely(model)")
        print("\nThen retrain for 50-100 epochs with normal training loop.")