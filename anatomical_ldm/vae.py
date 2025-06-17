"""
Anatomical-Aware VAE for Chest X-ray Latent Diffusion Models.
Extends diffusers AutoencoderKL with anatomical consistency features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import Attention
from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class AnatomicalVAE(AutoencoderKL):
    """
    Anatomical-Aware Variational Autoencoder for chest X-rays.
    Extends diffusers AutoencoderKL with anatomical consistency features.
    """
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        down_block_types: Tuple[str] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types: Tuple[str] = (
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 512,
        scaling_factor: float = 0.18215,
        # Anatomical-specific parameters
        num_anatomical_regions: int = 12,
        anatomical_loss_weight: float = 0.1,
    ):
        # Initialize parent AutoencoderKL
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
        )
        
        # Additional anatomical components
        self.num_anatomical_regions = num_anatomical_regions
        self.anatomical_loss_weight = anatomical_loss_weight
        
        # Anatomical attention in bottleneck
        bottleneck_channels = block_out_channels[-1]
        self.anatomical_attention = Attention(
            query_dim=bottleneck_channels,
            heads=8,
            dim_head=64,
            dropout=0.0,
        )
        
        # Anatomical segmentation head for auxiliary loss
        self.anatomical_head = nn.Sequential(
            nn.Conv2d(bottleneck_channels, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, num_anatomical_regions, 1),
        )
        
        # Hook to intercept encoder bottleneck features
        self._anatomical_features = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture anatomical features."""
        
        def bottleneck_hook(module, input, output):
            # Apply anatomical attention to encoder bottleneck
            batch_size, channels, height, width = output.shape
            
            # Flatten for attention
            h_flat = output.view(batch_size, channels, height * width).transpose(1, 2)
            h_attended = self.anatomical_attention(h_flat)
            h_attended = h_attended.transpose(1, 2).view(batch_size, channels, height, width)
            
            # Residual connection with small weight
            enhanced_output = output + h_attended * 0.1
            
            # Extract anatomical features
            self._anatomical_features = self.anatomical_head(enhanced_output)
            
            return enhanced_output
        
        # Hook into encoder mid_block
        self.encoder.mid_block.register_forward_hook(bottleneck_hook)
    
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Encode with anatomical feature extraction.
        
        Args:
            x: Input tensor [B, 1, H, W]
            return_dict: Whether to return dict
            
        Returns:
            Encoding output with anatomical features
        """
        # Reset anatomical features
        self._anatomical_features = None
        
        # Standard encoding (triggers our hook)
        posterior = super().encode(x, return_dict=return_dict)
        
        if return_dict:
            # Add anatomical features to output
            if hasattr(posterior, 'latent_dist'):
                # Handle AutoencoderKLOutput
                result = {
                    "latent_dist": posterior.latent_dist,
                    "anatomical_features": self._anatomical_features,
                }
            else:
                # Handle dict output
                result = dict(posterior)
                result["anatomical_features"] = self._anatomical_features
            
            return result
        else:
            return posterior
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Forward pass with anatomical consistency.
        
        Args:
            sample: Input image [B, 1, H, W]
            sample_posterior: Whether to sample from posterior
            return_dict: Whether to return dict
            generator: Random generator
            
        Returns:
            VAE output with anatomical features
        """
        # Standard VAE forward
        output = super().forward(
            sample=sample,
            sample_posterior=sample_posterior,
            return_dict=return_dict,
            generator=generator,
        )
        
        if return_dict:
            if hasattr(output, 'sample'):
                # Handle AutoencoderKLOutput
                result = {
                    "sample": output.sample,
                    "anatomical_features": self._anatomical_features,
                }
                if hasattr(output, 'latent_dist'):
                    result["latent_dist"] = output.latent_dist
            else:
                # Handle dict output
                result = dict(output)
                result["anatomical_features"] = self._anatomical_features
            
            return result
        else:
            return output
    
    def get_anatomical_consistency_loss(
        self,
        anatomical_features: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anatomical consistency loss.
        
        Args:
            anatomical_features: Predicted anatomical regions [B, num_regions, H, W]
            target_masks: Optional target segmentation masks [B, num_regions, H, W]
            
        Returns:
            Anatomical consistency loss
        """
        if target_masks is not None:
            # Supervised anatomical loss
            target_masks = F.interpolate(
                target_masks.float(),
                size=anatomical_features.shape[-2:],
                mode='nearest'
            )
            return F.cross_entropy(anatomical_features, target_masks.long())
        else:
            # Unsupervised anatomical regularization
            # Encourage spatial coherence and separation
            probs = F.softmax(anatomical_features, dim=1)
            
            # Entropy regularization (encourage confident predictions)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropy_loss = entropy.mean()
            
            # Spatial smoothness (encourage coherent regions)
            spatial_diff_h = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
            spatial_diff_w = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
            smoothness_loss = spatial_diff_h.mean() + spatial_diff_w.mean()
            
            return entropy_loss + 0.1 * smoothness_loss
    
    def compute_loss(
        self,
        sample: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full VAE loss with anatomical consistency.
        
        Args:
            sample: Input images [B, 1, H, W]
            target_masks: Optional anatomical masks [B, num_regions, H, W]
            beta: KL loss weight
            
        Returns:
            Dict of loss components
        """
        # Forward pass
        output = self.forward(sample, sample_posterior=True, return_dict=True)
        reconstruction = output["sample"]
        anatomical_features = output["anatomical_features"]
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, sample)
        
        # KL loss
        if hasattr(output, "latent_dist"):
            kl_loss = output.latent_dist.kl().mean()
        else:
            # Fallback KL computation
            posterior = self.encode(sample, return_dict=True)["latent_dist"]
            kl_loss = posterior.kl().mean()
        
        # Anatomical consistency loss
        anatomical_loss = self.get_anatomical_consistency_loss(
            anatomical_features, target_masks
        )
        
        # Total loss
        total_loss = (
            recon_loss +
            beta * kl_loss +
            self.anatomical_loss_weight * anatomical_loss
        )
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "anatomical_loss": anatomical_loss,
        }


# Factory function for easy instantiation
def create_anatomical_vae(
    image_size: int = 512,
    latent_channels: int = 4,
    num_anatomical_regions: int = 12,
    **kwargs
) -> AnatomicalVAE:
    """
    Create anatomical VAE with sensible defaults for chest X-rays.
    
    Args:
        image_size: Input image size
        latent_channels: Number of latent channels
        num_anatomical_regions: Number of anatomical regions to predict
        **kwargs: Additional config arguments
        
    Returns:
        Configured AnatomicalVAE
    """
    # Determine architecture based on image size
    if image_size >= 512:
        down_blocks = ("DownEncoderBlock2D",) * 4  # 8x downsampling
        block_channels = (128, 256, 512, 512)
    elif image_size >= 256:
        down_blocks = ("DownEncoderBlock2D",) * 3  # 4x downsampling  
        block_channels = (128, 256, 512)
    else:
        down_blocks = ("DownEncoderBlock2D",) * 2  # 2x downsampling
        block_channels = (128, 256)
    
    up_blocks = tuple("UpDecoderBlock2D" for _ in range(len(down_blocks)))
    
    return AnatomicalVAE(
        in_channels=1,  # Grayscale chest X-rays
        out_channels=1,
        latent_channels=latent_channels,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        block_out_channels=block_channels,
        sample_size=image_size,
        num_anatomical_regions=num_anatomical_regions,
        **kwargs
    )