#!/usr/bin/env python3
"""
Simple test to verify anatomical registers implementation works correctly.
"""

import torch
import torch.nn as nn
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

def test_anatomical_registers():
    """Test anatomical register functionality."""
    print("Testing Anatomical Registers Implementation...")
    
    # Test 1: AnatomicalRegisterBank
    print("1. Testing AnatomicalRegisterBank...")
    register_bank = AnatomicalRegisterBank(dim=512, num_organ_registers=12, num_spatial_registers=6, num_scale_registers=4)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    register_dict = register_bank(x, timesteps)
    print(f"   Register shapes: {register_dict['registers'].shape}")
    assert register_dict['registers'].shape == (batch_size, 22, 512)  # 12+6+4 = 22 registers
    
    # Test 2: RegisterModulatedUNet
    print("2. Testing RegisterModulatedUNet...")
    
    # Create a simple UNet model
    unet = diffusers.UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    
    # Wrap with anatomical registers
    modulated_unet = RegisterModulatedUNet(unet, register_bank)
    
    # Test forward pass
    output = modulated_unet(x, timesteps, return_dict=True)
    print(f"   Output shape: {output.sample.shape}")
    assert output.sample.shape == (batch_size, 1, 256, 256)
    
    # Test 3: Device handling
    print("3. Testing device handling...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        modulated_unet = modulated_unet.to(device)
        x = x.to(device)
        timesteps = timesteps.to(device)
        
        output = modulated_unet(x, timesteps, return_dict=True)
        assert output.sample.device == device
        print("   CUDA test passed")
    else:
        print("   CUDA not available, skipping GPU test")
    
    # Test 4: Stage-aware behavior
    print("4. Testing stage-aware behavior...")
    
    # Layout stage (high timesteps)
    high_timesteps = torch.full((batch_size,), 800)
    register_dict_layout = register_bank(x.cpu(), high_timesteps)
    
    # Detail stage (low timesteps)  
    low_timesteps = torch.full((batch_size,), 200)
    register_dict_detail = register_bank(x.cpu(), low_timesteps)
    
    # Registers should be different between stages
    layout_registers = register_dict_layout['registers']
    detail_registers = register_dict_detail['registers']
    
    assert not torch.allclose(layout_registers, detail_registers, atol=1e-5)
    print("   Stage-aware behavior verified")
    
    print("✅ All tests passed! Anatomical registers implementation is correct.")

if __name__ == "__main__":
    test_anatomical_registers()