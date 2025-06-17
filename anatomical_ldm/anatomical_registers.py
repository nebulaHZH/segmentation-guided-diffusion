"""
Anatomical Register Bank for LDM cross-attention conditioning.
Provides organ-specific, spatially-aware, and stage-aware anatomical guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for spatial anatomical awareness.
    """
    
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class SpatialPositionEmbedding(nn.Module):
    """
    2D spatial position embeddings for anatomical layout awareness.
    """
    
    def __init__(self, d_model: int, max_height: int = 64, max_width: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # Create 2D positional embeddings
        self.height_embedding = nn.Embedding(max_height, d_model // 2)
        self.width_embedding = nn.Embedding(max_width, d_model // 2)
        
    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D positional embeddings.
        
        Args:
            height: Height dimension
            width: Width dimension
            device: Target device
            
        Returns:
            Position embeddings [H*W, d_model]
        """
        h_pos = torch.arange(height, device=device)
        w_pos = torch.arange(width, device=device)
        
        h_embed = self.height_embedding(h_pos)  # [H, d_model//2]
        w_embed = self.width_embedding(w_pos)   # [W, d_model//2]
        
        # Create grid
        h_grid = h_embed.unsqueeze(1).expand(-1, width, -1)  # [H, W, d_model//2]
        w_grid = w_embed.unsqueeze(0).expand(height, -1, -1)  # [H, W, d_model//2]
        
        # Concatenate
        pos_embed = torch.cat([h_grid, w_grid], dim=-1)  # [H, W, d_model]
        
        return pos_embed.view(-1, self.d_model)  # [H*W, d_model]


class TimestepAwareEmbedding(nn.Module):
    """
    Timestep-aware embeddings for stage-based anatomical conditioning.
    """
    
    def __init__(self, d_model: int, max_timestep: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_timestep = max_timestep
        
        # Learnable timestep embedding
        self.timestep_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )
        
        # Stage classification (layout vs detail)
        self.stage_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),  # Early, Middle, Late stage
            nn.Softmax(dim=-1)
        )
    
    def forward(self, timestep: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate timestep-aware embeddings.
        
        Args:
            timestep: Timestep tensor [B]
            
        Returns:
            Dict with timestep embeddings and stage weights
        """
        # Normalize timestep
        t_norm = timestep.float() / self.max_timestep  # [B]
        
        # Project to embedding space
        t_embed = self.timestep_proj(t_norm.unsqueeze(-1))  # [B, d_model]
        
        # Classify stage
        stage_weights = self.stage_classifier(t_embed)  # [B, 3]
        
        return {
            "timestep_embed": t_embed,
            "stage_weights": stage_weights,  # [early, middle, late]
        }


class AnatomicalRegisterBank(nn.Module):
    """
    Bank of anatomical registers providing organ-specific, spatial, and temporal conditioning.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_organs: int = 12,
        spatial_resolution: int = 8,  # Latent space resolution
        max_timestep: int = 1000,
        organ_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_organs = num_organs
        self.spatial_resolution = spatial_resolution
        
        # Default organ names for chest X-rays
        if organ_names is None:
            self.organ_names = [
                "heart", "left_lung", "right_lung", "liver", "stomach",
                "left_ribs", "right_ribs", "spine", "clavicle", "diaphragm",
                "mediastinum", "background"
            ][:num_organs]
        else:
            self.organ_names = organ_names
        
        # Learnable organ embeddings
        self.organ_embeddings = nn.Parameter(torch.randn(num_organs, d_model))
        
        # Spatial position embeddings
        self.spatial_pos_embed = SpatialPositionEmbedding(
            d_model, spatial_resolution, spatial_resolution
        )
        
        # Timestep-aware embeddings
        self.timestep_embed = TimestepAwareEmbedding(d_model, max_timestep)
        
        # Stage-specific organ modulation
        self.stage_organ_weights = nn.ModuleList([
            nn.Linear(d_model, num_organs) for _ in range(3)  # early, middle, late
        ])
        
        # Anatomical layout priors (learnable spatial attention)
        self.layout_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cross-organ interaction
        self.organ_interaction = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection for cross-attention conditioning
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize anatomical register weights with anatomical priors."""
        
        # Initialize organ embeddings with small random values
        nn.init.xavier_uniform_(self.organ_embeddings, gain=0.1)
        
        # Initialize stage weights to reasonable defaults
        with torch.no_grad():
            # Early stage: focus on layout organs (heart, lungs, spine)
            self.stage_organ_weights[0].weight.fill_(0.1)
            self.stage_organ_weights[0].weight[:, [0, 1, 2, 7]] = 1.0  # heart, lungs, spine
            
            # Middle stage: balanced attention
            self.stage_organ_weights[1].weight.fill_(0.5)
            
            # Late stage: focus on detail organs (ribs, clavicle)
            self.stage_organ_weights[2].weight.fill_(0.1)
            self.stage_organ_weights[2].weight[:, [5, 6, 8]] = 1.0  # ribs, clavicle
    
    def forward(
        self,
        batch_size: int,
        timestep: torch.Tensor,
        device: torch.device,
        latent_height: int = 8,
        latent_width: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate anatomical conditioning features.
        
        Args:
            batch_size: Batch size
            timestep: Diffusion timestep [B]
            device: Target device
            latent_height: Height of latent space
            latent_width: Width of latent space
            
        Returns:
            Dict containing anatomical conditioning features
        """
        
        # Get timestep-aware embeddings
        time_embed_dict = self.timestep_embed(timestep)
        timestep_embed = time_embed_dict["timestep_embed"]  # [B, d_model]
        stage_weights = time_embed_dict["stage_weights"]    # [B, 3]
        
        # Get spatial position embeddings
        spatial_embed = self.spatial_pos_embed(
            latent_height, latent_width, device
        )  # [H*W, d_model]
        spatial_embed = spatial_embed.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, H*W, d_model]
        
        # Get organ embeddings
        organ_embed = self.organ_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_organs, d_model]
        
        # Apply stage-aware organ weighting
        stage_modulated_organs = []
        for stage_idx in range(3):
            stage_weight = stage_weights[:, stage_idx:stage_idx+1]  # [B, 1]
            organ_weights = torch.sigmoid(
                self.stage_organ_weights[stage_idx](timestep_embed)
            )  # [B, num_organs]
            
            weighted_organs = organ_embed * organ_weights.unsqueeze(-1)  # [B, num_organs, d_model]
            stage_modulated_organs.append(weighted_organs * stage_weight.unsqueeze(-1))
        
        # Combine stage-modulated organs
        modulated_organs = sum(stage_modulated_organs)  # [B, num_organs, d_model]
        
        # Add timestep information to organs
        timestep_broadcast = timestep_embed.unsqueeze(1).expand(-1, self.num_organs, -1)
        organ_with_time = modulated_organs + timestep_broadcast * 0.1
        
        # Cross-organ interaction
        interacted_organs = self.organ_interaction(organ_with_time)  # [B, num_organs, d_model]
        
        # Generate spatial anatomical features via attention
        # Each organ attends to spatial positions
        spatial_anatomical_features = []
        
        for organ_idx in range(self.num_organs):
            organ_query = interacted_organs[:, organ_idx:organ_idx+1, :]  # [B, 1, d_model]
            
            # Attention between organ and spatial positions
            attended_spatial, spatial_attn_weights = self.layout_attention(
                query=organ_query,
                key=spatial_embed,
                value=spatial_embed,
            )  # [B, 1, d_model], [B, 1, H*W]
            
            spatial_anatomical_features.append(attended_spatial)
        
        # Combine all organ-spatial features
        spatial_anatomical = torch.cat(spatial_anatomical_features, dim=1)  # [B, num_organs, d_model]
        
        # Final projection for cross-attention
        conditioning_features = self.output_proj(spatial_anatomical)  # [B, num_organs, d_model]
        
        # Create expanded spatial-organ features for dense conditioning
        # Repeat each organ for each spatial location
        dense_features = []
        for organ_idx in range(self.num_organs):
            organ_feat = conditioning_features[:, organ_idx:organ_idx+1, :].expand(
                -1, latent_height * latent_width, -1
            )  # [B, H*W, d_model]
            dense_features.append(organ_feat)
        
        dense_conditioning = torch.stack(dense_features, dim=2)  # [B, H*W, num_organs, d_model]
        dense_conditioning = dense_conditioning.view(
            batch_size, latent_height * latent_width * self.num_organs, self.d_model
        )  # [B, H*W*num_organs, d_model]
        
        return {
            "organ_conditioning": conditioning_features,        # [B, num_organs, d_model]
            "spatial_conditioning": spatial_embed,             # [B, H*W, d_model]  
            "dense_conditioning": dense_conditioning,          # [B, H*W*num_organs, d_model]
            "timestep_embed": timestep_embed,                  # [B, d_model]
            "stage_weights": stage_weights,                    # [B, 3]
            "organ_names": self.organ_names,
        }


class AnatomicalConditioningAdapter(nn.Module):
    """
    Adapter to integrate anatomical conditioning into existing attention layers.
    """
    
    def __init__(
        self,
        cross_attention_dim: int,
        anatomical_dim: int = 512,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
    ):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.anatomical_dim = anatomical_dim
        
        # Project anatomical features to match cross-attention dimension
        self.anatomical_proj = nn.Linear(anatomical_dim, cross_attention_dim)
        
        # Attention mechanism for blending text and anatomical conditioning
        self.conditioning_attention = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating mechanism to control anatomical influence
        self.anatomical_gate = nn.Sequential(
            nn.Linear(cross_attention_dim * 2, cross_attention_dim),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        text_conditioning: Optional[torch.Tensor],
        anatomical_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine text and anatomical conditioning.
        
        Args:
            text_conditioning: Text embeddings [B, text_seq_len, cross_attention_dim]
            anatomical_conditioning: Anatomical features [B, anat_seq_len, anatomical_dim]
            
        Returns:
            Combined conditioning [B, total_seq_len, cross_attention_dim]
        """
        batch_size = anatomical_conditioning.shape[0]
        
        # Project anatomical features
        anat_proj = self.anatomical_proj(anatomical_conditioning)  # [B, anat_seq_len, cross_attention_dim]
        
        if text_conditioning is not None:
            # Combine text and anatomical
            combined_seq = torch.cat([text_conditioning, anat_proj], dim=1)
            
            # Attention-based fusion
            fused_features, _ = self.conditioning_attention(
                query=combined_seq,
                key=combined_seq, 
                value=combined_seq,
            )
            
            # Gating to control anatomical influence
            text_features = fused_features[:, :text_conditioning.shape[1]]
            anat_features = fused_features[:, text_conditioning.shape[1]:]
            
            # Compute gate
            gate_input = torch.cat([
                text_features.mean(dim=1), 
                anat_features.mean(dim=1)
            ], dim=1)
            gate = self.anatomical_gate(gate_input).unsqueeze(1)  # [B, 1, 1]
            
            # Apply gate
            final_conditioning = text_features * (1 - gate) + anat_features * gate
            
            return torch.cat([final_conditioning, anat_features], dim=1)
        else:
            # Only anatomical conditioning
            return anat_proj


# Factory function
def create_anatomical_registers(
    latent_resolution: int = 8,
    d_model: int = 512,
    num_organs: int = 12,
    **kwargs
) -> AnatomicalRegisterBank:
    """
    Create anatomical register bank with sensible defaults.
    
    Args:
        latent_resolution: Resolution of latent space
        d_model: Model dimension
        num_organs: Number of anatomical organs
        **kwargs: Additional arguments
        
    Returns:
        Configured AnatomicalRegisterBank
    """
    return AnatomicalRegisterBank(
        d_model=d_model,
        num_organs=num_organs,
        spatial_resolution=latent_resolution,
        **kwargs
    )