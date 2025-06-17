"""
Anatomical Cross-Attention UNet for Latent Diffusion.
Extends diffusers UNet2DConditionModel with anatomical conditioning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Any
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging

from .anatomical_registers import AnatomicalRegisterBank, AnatomicalConditioningAdapter

logger = logging.get_logger(__name__)


class AnatomicalAttnProcessor(AttnProcessor):
    """
    Attention processor that includes anatomical cross-attention.
    """
    
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        anatomical_dim: int = 512,
        anatomical_heads: int = 8,
        anatomical_head_dim: int = 64,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.anatomical_dim = anatomical_dim
        
        # Anatomical cross-attention layer
        self.anatomical_attention = Attention(
            query_dim=hidden_size,
            cross_attention_dim=anatomical_dim,
            heads=anatomical_heads,
            dim_head=anatomical_head_dim,
            dropout=0.1,
        )
        
        # Gating mechanism to control anatomical influence
        self.anatomical_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Layer norm for anatomical features
        self.norm_anatomical = nn.LayerNorm(hidden_size)
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        anatomical_conditioning: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.FloatTensor:
        """
        Apply attention with anatomical conditioning.
        
        Args:
            attn: Base attention layer
            hidden_states: Input features [B, H*W, C]
            encoder_hidden_states: Text conditioning [B, text_seq_len, text_dim]
            attention_mask: Attention mask
            anatomical_conditioning: Anatomical features [B, anat_seq_len, anat_dim]
            
        Returns:
            Attention output with anatomical conditioning
        """
        residual = hidden_states
        
        # Standard cross-attention (text conditioning)
        if encoder_hidden_states is not None:
            # Use the base attention processor
            text_attended = super(AnatomicalAttnProcessor, self).__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs
            )
        else:
            # Self-attention
            text_attended = super(AnatomicalAttnProcessor, self).__call__(
                attn, hidden_states, None, attention_mask, **cross_attention_kwargs
            )
        
        # Anatomical cross-attention
        if anatomical_conditioning is not None:
            # Apply anatomical cross-attention
            anatomical_attended = self.anatomical_attention(
                hidden_states, anatomical_conditioning
            )
            
            # Compute gate to control anatomical influence
            gate = self.anatomical_gate(hidden_states.mean(dim=1, keepdim=True))  # [B, 1, 1]
            
            # Combine text and anatomical attention
            combined_output = text_attended * (1 - gate) + anatomical_attended * gate
            
            # Residual connection with normalization
            output = residual + self.norm_anatomical(combined_output)
        else:
            output = text_attended
        
        return output


class AnatomicalUNet2DConditionModel(UNet2DConditionModel):
    """
    UNet with anatomical cross-attention conditioning.
    Extends diffusers UNet2DConditionModel.
    """
    
    @register_to_config
    def __init__(
        self,
        # Standard UNet2DConditionModel parameters
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
        
        # Anatomical conditioning parameters
        anatomical_conditioning_dim: int = 512,
        anatomical_num_organs: int = 12,
        anatomical_attention_heads: int = 8,
        anatomical_head_dim: int = 64,
        anatomical_layers: Optional[Tuple[int]] = None,  # Which layers to add anatomical conditioning
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )
        
        # Store anatomical conditioning parameters
        self.anatomical_conditioning_dim = anatomical_conditioning_dim
        self.anatomical_num_organs = anatomical_num_organs
        
        # Anatomical register bank
        self.anatomical_registers = AnatomicalRegisterBank(
            d_model=anatomical_conditioning_dim,
            num_organs=anatomical_num_organs,
            spatial_resolution=8,  # Assuming 64x64 latent -> 8x8 at bottleneck
        )
        
        # Conditioning adapter for text + anatomical fusion
        self.conditioning_adapter = AnatomicalConditioningAdapter(
            cross_attention_dim=cross_attention_dim,
            anatomical_dim=anatomical_conditioning_dim,
            num_attention_heads=anatomical_attention_heads,
            attention_head_dim=anatomical_head_dim,
        )
        
        # Determine which layers to add anatomical conditioning
        if anatomical_layers is None:
            # Default: add to cross-attention layers
            self.anatomical_layers = list(range(len(down_block_types) + len(up_block_types) + 1))
        else:
            self.anatomical_layers = anatomical_layers
        
        # Replace attention processors with anatomical versions
        self._setup_anatomical_attention_processors()
    
    def _setup_anatomical_attention_processors(self):
        """
        Replace attention processors with anatomical versions.
        """
        attention_processors = {}
        
        def setup_processors(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if hasattr(child, "set_processor"):
                    # This is an attention layer
                    if hasattr(child, "cross_attention_dim") and child.cross_attention_dim is not None:
                        # Cross-attention layer - add anatomical conditioning
                        attention_processors[full_name] = AnatomicalAttnProcessor(
                            hidden_size=child.inner_dim,
                            cross_attention_dim=child.cross_attention_dim,
                            anatomical_dim=self.anatomical_conditioning_dim,
                            anatomical_heads=8,
                            anatomical_head_dim=64,
                        )
                    else:
                        # Self-attention layer - use default processor
                        attention_processors[full_name] = AttnProcessor()
                else:
                    # Recursively process child modules
                    setup_processors(child, full_name)
        
        setup_processors(self)
        self.set_attn_processor(attention_processors)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass with anatomical conditioning.
        
        Args:
            sample: Noisy latent tensor [B, C, H, W]
            timestep: Diffusion timestep
            encoder_hidden_states: Text conditioning [B, text_seq_len, text_dim] 
            ... (other standard UNet parameters)
            
        Returns:
            UNet output with anatomical conditioning applied
        """
        batch_size, channels, height, width = sample.shape
        device = sample.device
        
        # Ensure timestep is tensor
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep] * batch_size, device=device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size).to(device)
        
        # Generate anatomical conditioning
        anatomical_dict = self.anatomical_registers(
            batch_size=batch_size,
            timestep=timestep,
            device=device,
            latent_height=height,
            latent_width=width,
        )
        
        # Combine text and anatomical conditioning
        if encoder_hidden_states is not None:
            combined_conditioning = self.conditioning_adapter(
                text_conditioning=encoder_hidden_states,
                anatomical_conditioning=anatomical_dict["organ_conditioning"],
            )
        else:
            combined_conditioning = self.conditioning_adapter(
                text_conditioning=None,
                anatomical_conditioning=anatomical_dict["organ_conditioning"],
            )
        
        # Add anatomical conditioning to cross-attention kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        
        cross_attention_kwargs["anatomical_conditioning"] = anatomical_dict["dense_conditioning"]
        
        # Call parent forward with modified conditioning
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=combined_conditioning,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
    
    def set_anatomical_conditioning_scale(self, scale: float):
        """
        Set the scale of anatomical conditioning influence.
        
        Args:
            scale: Scale factor for anatomical conditioning (0.0 to 1.0)
        """
        for processor in self.attn_processors.values():
            if isinstance(processor, AnatomicalAttnProcessor):
                # Modify gate bias to control influence
                with torch.no_grad():
                    processor.anatomical_gate[-2].bias.fill_(scale - 0.5)


# Factory function
def create_anatomical_unet(
    image_size: int = 512,
    latent_channels: int = 4,
    cross_attention_dim: int = 768,
    anatomical_conditioning_dim: int = 512,
    num_organs: int = 12,
    **kwargs
) -> AnatomicalUNet2DConditionModel:
    """
    Create anatomical UNet with sensible defaults.
    
    Args:
        image_size: Input image size
        latent_channels: Number of latent channels
        cross_attention_dim: Text conditioning dimension
        anatomical_conditioning_dim: Anatomical conditioning dimension
        num_organs: Number of anatomical organs
        **kwargs: Additional UNet arguments
        
    Returns:
        Configured AnatomicalUNet2DConditionModel
    """
    # Determine appropriate architecture based on image size
    if image_size >= 512:
        block_out_channels = (320, 640, 1280, 1280)
        down_block_types = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        up_block_types = (
            "UpBlock2D",
            "CrossAttnUpBlock2D", 
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        )
    else:
        block_out_channels = (128, 256, 512, 512)
        down_block_types = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        up_block_types = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        )
    
    return AnatomicalUNet2DConditionModel(
        sample_size=image_size // 8,  # Latent resolution
        in_channels=latent_channels,
        out_channels=latent_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        cross_attention_dim=cross_attention_dim,
        anatomical_conditioning_dim=anatomical_conditioning_dim,
        anatomical_num_organs=num_organs,
        **kwargs
    )