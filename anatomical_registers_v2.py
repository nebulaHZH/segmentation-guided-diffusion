"""
Improved Anatomical Register Bank with fixes for common training issues.
This version includes:
- Gate warmup to gradually introduce registers
- Better initialization strategies
- Minimum gate value to ensure registers are used
- Optional auxiliary losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple

# Import original components
from anatomical_registers import AnatomicalRegisterBank


class ImprovedGate(nn.Module):
    """Improved gate with warmup and minimum value."""
    
    def __init__(self, input_dim, min_value=0.1, max_value=0.5, warmup_steps=10000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        self.min_value = min_value
        self.max_value = max_value
        self.warmup_steps = warmup_steps
        self.register_buffer('step_count', torch.tensor(0))
        
    def forward(self, x):
        # Base gate value
        gate = self.net(x)
        
        # Apply warmup during training
        if self.training:
            warmup_factor = torch.clamp(self.step_count.float() / self.warmup_steps, 0, 1)
            self.step_count += 1
        else:
            warmup_factor = 1.0
            
        # Scale to [min_value, max_value] range with warmup
        gate = self.min_value + (self.max_value - self.min_value) * gate * warmup_factor
        
        return gate


class RegisterModulatedUNetV2(nn.Module):
    """
    Improved wrapper around UNet2DModel with better register integration.
    
    Key improvements:
    - Gate warmup to gradually introduce registers
    - Minimum gate value to ensure registers are always used
    - Better initialization options
    - Optional intermediate feature modulation
    """
    
    def __init__(
        self, 
        unet_model, 
        register_bank: AnatomicalRegisterBank,
        min_gate=0.1,
        max_gate=0.5,
        warmup_steps=10000,
        modulate_intermediate=False
    ):
        super().__init__()
        self.unet = unet_model
        self.register_bank = register_bank
        self.modulate_intermediate = modulate_intermediate
        
        # Get UNet dimensions
        self.in_channels = self.unet.config.in_channels
        self.out_channels = self.unet.config.out_channels
        
        # Improved projection and gate
        self.register_proj = nn.Linear(register_bank.dim, self.out_channels)
        self.gate = ImprovedGate(
            register_bank.dim, 
            min_value=min_gate,
            max_value=max_gate,
            warmup_steps=warmup_steps
        )
        
        # Initialize projection with small values to start
        nn.init.xavier_uniform_(self.register_proj.weight, gain=0.1)
        
        # Copy UNet attributes
        self.config = self.unet.config
        
        # For intermediate modulation
        if modulate_intermediate:
            self.intermediate_projs = nn.ModuleDict()
            self.intermediate_gates = nn.ModuleDict()
            self._setup_intermediate_modulation()
    
    def _setup_intermediate_modulation(self):
        """Setup projections for intermediate feature modulation."""
        # Setup for down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, 'resnets'):
                channels = down_block.resnets[0].out_channels
                self.intermediate_projs[f'down_{i}'] = nn.Linear(
                    self.register_bank.dim, channels
                )
                self.intermediate_gates[f'down_{i}'] = ImprovedGate(
                    self.register_bank.dim, min_value=0.05, max_value=0.3
                )
        
        # Setup for up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, 'resnets'):
                channels = up_block.resnets[0].out_channels
                self.intermediate_projs[f'up_{i}'] = nn.Linear(
                    self.register_bank.dim, channels
                )
                self.intermediate_gates[f'up_{i}'] = ImprovedGate(
                    self.register_bank.dim, min_value=0.05, max_value=0.3
                )
    
    @property 
    def dtype(self):
        return self.unet.dtype
        
    @property
    def device(self):
        return next(self.unet.parameters()).device
        
    def forward(self, x, timestep, class_labels=None, return_dict=True):
        """Forward pass with improved register modulation."""
        # Get anatomical registers
        register_dict = self.register_bank(x, timestep)
        registers = register_dict["registers"]
        
        # Pool registers
        pooled_registers = registers.mean(dim=1)  # [B, register_dim]
        
        if self.modulate_intermediate and self.training:
            # Modulate intermediate features during training
            output = self._forward_with_intermediate_modulation(
                x, timestep, pooled_registers, class_labels
            )
        else:
            # Standard forward with output modulation only
            output = self._forward_with_output_modulation(
                x, timestep, pooled_registers, class_labels
            )
        
        if return_dict:
            from collections import namedtuple
            UNet2DOutput = namedtuple('UNet2DOutput', ['sample'])
            return UNet2DOutput(sample=output)
        else:
            return (output,)
    
    def _forward_with_output_modulation(self, x, timestep, pooled_registers, class_labels):
        """Standard forward pass with output-only modulation."""
        # Get UNet output
        unet_output = self.unet(x, timestep, class_labels=class_labels, return_dict=True)
        sample = unet_output.sample
        
        # Project registers to output dimension
        register_features = self.register_proj(pooled_registers)  # [B, out_channels]
        
        # Compute gate
        gate_value = self.gate(pooled_registers)  # [B, 1]
        
        # Apply modulation
        B, C, H, W = sample.shape
        register_features = register_features.view(B, C, 1, 1).expand(-1, -1, H, W)
        gate_value = gate_value.view(B, 1, 1, 1).expand(-1, C, H, W)
        
        modulated_sample = sample * (1 - gate_value) + register_features * gate_value
        
        return modulated_sample
    
    def _forward_with_intermediate_modulation(self, x, timestep, pooled_registers, class_labels):
        """Forward pass with intermediate feature modulation."""
        # This would require modifying the UNet forward pass
        # For now, fall back to output modulation
        return self._forward_with_output_modulation(x, timestep, pooled_registers, class_labels)
    
    def get_register_loss(self, x, timestep, alpha=0.01):
        """
        Auxiliary loss to encourage register usage.
        Returns a loss that encourages the model to use registers meaningfully.
        """
        # Get current output
        with torch.no_grad():
            current_output = self.forward(x, timestep, return_dict=True).sample
        
        # Get output with minimal register influence
        old_min = self.gate.min_value
        self.gate.min_value = 0.0
        minimal_output = self.forward(x, timestep, return_dict=True).sample
        self.gate.min_value = old_min
        
        # Encourage difference
        register_diff = torch.mean(torch.abs(current_output - minimal_output))
        
        # Also get current gate values
        register_dict = self.register_bank(x, timestep)
        pooled = register_dict["registers"].mean(dim=1)
        gate_values = self.gate(pooled)
        
        # Encourage non-zero gates (but not too high)
        gate_loss = -torch.log(gate_values + 1e-8).mean() + torch.relu(gate_values - 0.7).mean()
        
        return alpha * (gate_loss - register_diff)


def create_improved_model(base_unet, config):
    """
    Create an improved anatomical register model with better defaults.
    
    Args:
        base_unet: The base UNet2DModel
        config: Training configuration
        
    Returns:
        RegisterModulatedUNetV2 model
    """
    # Create register bank
    register_bank = AnatomicalRegisterBank(
        dim=config.register_dim,
        num_organ_registers=config.num_organ_registers,
        num_spatial_registers=config.num_spatial_registers,
        num_scale_registers=config.num_scale_registers
    )
    
    # Calculate warmup steps based on training
    warmup_epochs = min(50, config.num_epochs // 4)
    steps_per_epoch = 1000  # Approximate
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Create improved model
    model = RegisterModulatedUNetV2(
        base_unet,
        register_bank,
        min_gate=0.1,      # Minimum 10% register influence
        max_gate=0.5,      # Maximum 50% register influence  
        warmup_steps=warmup_steps,
        modulate_intermediate=False  # Can enable for stronger integration
    )
    
    return model


# Training modifications to use auxiliary loss
def train_step_with_register_loss(model, noisy_images, noise, timesteps, base_loss_weight=1.0, register_loss_weight=0.01):
    """
    Modified training step that includes register loss.
    
    Example usage in training loop:
    ```python
    # Instead of:
    # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
    # loss = F.mse_loss(noise_pred, noise)
    
    # Use:
    loss = train_step_with_register_loss(
        model, noisy_images, noise, timesteps,
        base_loss_weight=1.0, register_loss_weight=0.01
    )
    ```
    """
    # Get noise prediction
    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
    
    # Base diffusion loss
    base_loss = F.mse_loss(noise_pred, noise)
    
    # Add register loss if using improved model
    if isinstance(model.module if hasattr(model, 'module') else model, RegisterModulatedUNetV2):
        register_loss = model.module.get_register_loss(noisy_images, timesteps)
        total_loss = base_loss_weight * base_loss + register_loss_weight * register_loss
    else:
        total_loss = base_loss
    
    return total_loss