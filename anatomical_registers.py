"""
Anatomical Register Bank for medical image generation.
Minimal implementation that integrates with existing segmentation-guided diffusion codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List


class AnatomicalRegisterBank(nn.Module):
    """Learnable anatomical register tokens for medical image generation."""
    
    def __init__(
        self,
        dim: int = 512,  # Match UNet's hidden dimension
        num_organ_registers: int = 12,
        num_spatial_registers: int = 6,
        num_scale_registers: int = 4,
        organ_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_organ_registers = num_organ_registers
        self.num_spatial_registers = num_spatial_registers
        self.num_scale_registers = num_scale_registers
        
        # Initialize register tokens
        self.organ_registers = nn.Parameter(torch.randn(num_organ_registers, dim))
        self.spatial_registers = nn.Parameter(torch.randn(num_spatial_registers, dim))
        self.scale_registers = nn.Parameter(torch.randn(num_scale_registers, dim))
        
        # Learnable mixing parameters
        self.register_mixing = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.organ_registers)
        nn.init.xavier_uniform_(self.spatial_registers)
        nn.init.xavier_uniform_(self.scale_registers)
        
        # Register names for interpretability
        self.organ_names = organ_names or [
            "heart", "lung_left", "lung_right", "liver", "spleen", 
            "kidney_left", "kidney_right", "spine", "ribs", "muscle",
            "fat", "background"
        ][:num_organ_registers]
        
    def get_stage_registers(self, timestep: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Get registers based on diffusion timestep (stage-aware)."""
        # Determine stage based on timestep
        if timestep is not None:
            timestep = timestep.to(self.organ_registers.device)
            threshold = 500.0
            layout_mask = timestep.float() > threshold
            
            if layout_mask.all().item():
                stage = "layout"
            elif not layout_mask.any().item():
                stage = "detail"
            else:
                num_layout = layout_mask.sum().item()
                stage = "layout" if num_layout > batch_size // 2 else "detail"
        else:
            stage = "all"
            
        # Get weighted registers based on stage
        weights = F.softmax(self.register_mixing, dim=0)
        
        if stage == "layout":
            # Early stage: emphasize organ and spatial registers
            registers = torch.cat([
                self.organ_registers * weights[0] * 2,
                self.spatial_registers * weights[1] * 2,
                self.scale_registers * weights[2] * 0.1
            ], dim=0)
        elif stage == "detail":
            # Late stage: emphasize scale registers
            registers = torch.cat([
                self.organ_registers * weights[0],
                self.spatial_registers * weights[1],
                self.scale_registers * weights[2] * 2
            ], dim=0)
        else:
            # All stages: balanced
            registers = torch.cat([
                self.organ_registers * weights[0],
                self.spatial_registers * weights[1],
                self.scale_registers * weights[2]
            ], dim=0)
            
        # Expand for batch
        registers = registers.unsqueeze(0).expand(batch_size, -1, -1)
        return registers
        
    def forward(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass returning register embeddings."""
        batch_size = x.shape[0]
        registers = self.get_stage_registers(timestep, batch_size)
        
        return {
            "registers": registers,
            "organ": self.organ_registers.unsqueeze(0).expand(batch_size, -1, -1),
            "spatial": self.spatial_registers.unsqueeze(0).expand(batch_size, -1, -1),
            "scale": self.scale_registers.unsqueeze(0).expand(batch_size, -1, -1),
        }


class RegisterModulatedUNet(nn.Module):
    """Wrapper around UNet2DModel to add anatomical register modulation."""
    
    def __init__(self, unet_model, register_bank: AnatomicalRegisterBank):
        super().__init__()
        self.unet = unet_model
        self.register_bank = register_bank
        
        # Get UNet's input and output dimensions
        self.in_channels = self.unet.config.in_channels
        self.out_channels = self.unet.config.out_channels
        
        # Projection layers to inject registers into UNet features
        self.register_proj = nn.Linear(register_bank.dim, self.out_channels)
        self.gate = nn.Sequential(
            nn.Linear(register_bank.dim, register_bank.dim // 4),
            nn.GELU(),
            nn.Linear(register_bank.dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Copy relevant attributes from the wrapped UNet for compatibility
        self.config = self.unet.config
        
    @property 
    def dtype(self):
        return self.unet.dtype
        
    @property
    def device(self):
        return next(self.unet.parameters()).device
        
    def forward(self, x, timestep, class_labels=None, return_dict=True):
        """Forward pass with register modulation."""
        # Get anatomical registers
        register_dict = self.register_bank(x, timestep)
        registers = register_dict["registers"]
        
        # Pool registers across the register dimension
        pooled_registers = registers.mean(dim=1)  # [B, register_dim]
        
        # Project registers to match UNet output dimension
        register_features = self.register_proj(pooled_registers)  # [B, out_channels]
        
        # Compute gating
        gate_value = self.gate(pooled_registers)  # [B, 1]
        
        # Get UNet features
        unet_output = self.unet(x, timestep, class_labels=class_labels, return_dict=True)
        
        # Apply register modulation to the output
        # This is a simplified version - in practice you'd modulate intermediate features
        if return_dict:
            sample = unet_output.sample
            # Apply gated modulation
            B, C, H, W = sample.shape
            
            # Ensure register_features matches output channels
            if register_features.shape[1] != C:
                # If dimensions don't match, pad or truncate
                if register_features.shape[1] < C:
                    # Pad with zeros
                    padding = torch.zeros(B, C - register_features.shape[1], device=register_features.device)
                    register_features = torch.cat([register_features, padding], dim=1)
                else:
                    # Truncate
                    register_features = register_features[:, :C]
            
            register_features = register_features.view(B, C, 1, 1).expand(-1, -1, H, W)
            gate_value = gate_value.view(B, 1, 1, 1).expand(-1, C, H, W)
            
            modulated_sample = sample * (1 - gate_value) + register_features * gate_value
            
            # Return in same format as original UNet
            from collections import namedtuple
            UNet2DOutput = namedtuple('UNet2DOutput', ['sample'])
            return UNet2DOutput(sample=modulated_sample)
        else:
            return unet_output