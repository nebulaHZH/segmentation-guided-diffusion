#!/usr/bin/env python3
"""
Test if the fundamental architecture approach can work by testing idealized scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from pathlib import Path
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
import diffusers

def test_ideal_registers(checkpoint_dir, device='cuda'):
    """Test if the architecture could work with ideal register features."""
    
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
    print("TESTING IF ARCHITECTURE CAN WORK WITH IDEAL FEATURES")
    print("="*60)
    
    # Test 1: Replace register features with meaningful image-based features
    print("\n1. TESTING WITH IMAGE-DERIVED FEATURES:")
    
    class ImageBasedRegisterBank(nn.Module):
        def __init__(self, original_bank):
            super().__init__()
            self.original_bank = original_bank
            
        def forward(self, x, timestep):
            # Instead of learned registers, use actual image statistics
            batch_size = x.shape[0]
            
            # Extract image features that could be "anatomical"
            mean_intensity = x.mean(dim=(2,3))  # Overall brightness
            std_intensity = x.std(dim=(2,3))    # Contrast
            
            # Edge detection (simple)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            edges = torch.conv2d(x, sobel_x, padding=1).abs().mean(dim=(2,3))
            
            # Spatial statistics
            center_intensity = x[:, :, 24:40, 24:40].mean(dim=(2,3))  # Center region
            border_intensity = x[:, :, :8, :].mean(dim=(2,3))         # Border region
            
            # Create "anatomical" features
            anatomical_features = torch.cat([
                mean_intensity, std_intensity, edges, 
                center_intensity, border_intensity
            ], dim=1)  # [B, 5]
            
            # Expand to match register dimension
            anatomical_features = anatomical_features.repeat(1, 512 // 5 + 1)[:, :512]
            
            # Create fake register dict
            return {
                "registers": anatomical_features.unsqueeze(1).expand(-1, 22, -1),
                "organ": anatomical_features.unsqueeze(1).expand(-1, 12, -1),
                "spatial": anatomical_features.unsqueeze(1).expand(-1, 6, -1),
                "scale": anatomical_features.unsqueeze(1).expand(-1, 4, -1),
            }
    
    # Replace register bank temporarily
    image_bank = ImageBasedRegisterBank(model.register_bank)
    original_bank = model.register_bank
    model.register_bank = image_bank
    
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64).to(device)
        t = torch.tensor([500]).to(device)
        
        # Test with original vs image-based features
        model.register_bank = original_bank
        original_out = model(x, t, return_dict=True).sample
        
        model.register_bank = image_bank
        image_out = model(x, t, return_dict=True).sample
        
        pure_unet = model.unet(x, t, return_dict=True).sample
        
        diff_original = torch.mean(torch.abs(original_out - pure_unet)).item()
        diff_image = torch.mean(torch.abs(image_out - pure_unet)).item()
        
        print(f"  Difference with learned registers: {diff_original:.6f}")
        print(f"  Difference with image-based registers: {diff_image:.6f}")
        
        if diff_image > diff_original:
            print("  ✓ Image-based features have more impact!")
        else:
            print("  ⚠️ Even image-based features don't help much")
    
    # Restore original bank
    model.register_bank = original_bank
    
    # Test 2: What if we modulated at a better location?
    print("\n2. TESTING INTERMEDIATE MODULATION:")
    
    # Hook into an intermediate layer
    def modulation_hook(module, input, output):
        # Get current registers
        # This is a simplified test - would need proper implementation
        register_influence = torch.randn_like(output) * 0.1  # Simulate register influence
        return output + register_influence
    
    # Try modulating different layers
    layers_to_test = [
        ("down_block_1", model.unet.down_blocks[1]),
        ("mid_block", model.unet.mid_block),
        ("up_block_1", model.unet.up_blocks[1]),
    ]
    
    baseline_out = model.unet(x, t, return_dict=True).sample
    
    for layer_name, layer in layers_to_test:
        with torch.no_grad():
            # Add hook
            hook = layer.register_forward_hook(modulation_hook)
            
            # Forward pass with modulation
            modulated_out = model.unet(x, t, return_dict=True).sample
            
            # Remove hook
            hook.remove()
            
            diff = torch.mean(torch.abs(modulated_out - baseline_out)).item()
            print(f"  {layer_name} modulation impact: {diff:.6f}")
    
    # Test 3: Scale sensitivity test
    print("\n3. TESTING SCALE SENSITIVITY:")
    
    with torch.no_grad():
        # Get current register features
        register_dict = model.register_bank(x, t)
        pooled = register_dict["registers"].mean(dim=1)
        projected = model.register_proj(pooled)
        
        # Test different scales
        scales = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        for scale in scales:
            # Manually create modulated output
            unet_out = model.unet(x, t, return_dict=True).sample
            scaled_features = projected * scale
            gate_val = 0.5  # Fixed gate for testing
            
            scaled_features = scaled_features.view(1, -1, 1, 1).expand_like(unet_out)
            manual_out = unet_out * (1 - gate_val) + scaled_features * gate_val
            
            diff = torch.mean(torch.abs(manual_out - unet_out)).item()
            print(f"  Scale {scale:6.2f}: difference = {diff:.6f}")
    
    # Test 4: What does the UNet think about different anatomical content?
    print("\n4. TESTING UNET ANATOMICAL SENSITIVITY:")
    
    # Create synthetic "anatomical" patterns
    patterns = {
        "empty": torch.zeros(1, 1, 64, 64),
        "center_blob": torch.zeros(1, 1, 64, 64),
        "dual_circles": torch.zeros(1, 1, 64, 64),
        "ribs": torch.zeros(1, 1, 64, 64),
    }
    
    # Center blob (heart-like)
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    center_mask = ((x - 32)**2 + (y - 40)**2) < 100
    patterns["center_blob"][0, 0][center_mask] = 1
    
    # Dual circles (lungs-like)
    left_lung = ((x - 20)**2 + (y - 32)**2) < 150
    right_lung = ((x - 44)**2 + (y - 32)**2) < 150
    patterns["dual_circles"][0, 0][left_lung | right_lung] = 1
    
    # Horizontal lines (ribs-like)
    for i in range(10, 55, 8):
        patterns["ribs"][0, 0, i:i+2, :] = 1
    
    # Add noise to make them more realistic
    for name, pattern in patterns.items():
        if name != "empty":
            pattern += torch.randn_like(pattern) * 0.1
        patterns[name] = pattern.to(device)
    
    # Test UNet response to different anatomical patterns
    with torch.no_grad():
        responses = {}
        for name, pattern in patterns.items():
            response = model.unet(pattern, t, return_dict=True).sample
            responses[name] = response
            print(f"  {name}: mean={response.mean().item():.4f}, std={response.std().item():.4f}")
        
        # Check if responses are meaningfully different
        baseline_resp = responses["empty"]
        for name, response in responses.items():
            if name != "empty":
                diff = torch.mean(torch.abs(response - baseline_resp)).item()
                print(f"    {name} vs empty: {diff:.6f}")
    
    print("\n" + "="*60)
    print("ARCHITECTURAL ASSESSMENT:")
    print("="*60)
    
    print("\nKey findings:")
    print("1. If image-based features don't help much → architecture is flawed")
    print("2. If intermediate modulation has little impact → timing is wrong")
    print("3. If scale sensitivity is extreme → combination method is wrong") 
    print("4. If UNet responses are similar → anatomical priors aren't needed")
    
    print(f"\nConclusion: The architecture fundamental issues are likely:")
    print("- Too late in pipeline (final output modulation)")
    print("- Wrong combination method (linear blend)")
    print("- Wrong feature space (register projection)")
    print("- UNet already learns anatomical patterns implicitly")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fundamental_architecture_test.py <checkpoint_dir>")
        sys.exit(1)
    
    test_ideal_registers(sys.argv[1])