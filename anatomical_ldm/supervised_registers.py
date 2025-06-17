"""
Supervised Anatomical Register Bank that learns from segmentation masks during training.
At test time, uses learned anatomical knowledge without requiring masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .anatomical_registers import SpatialPositionEmbedding, TimestepAwareEmbedding


class SupervisedAnatomicalRegisterBank(nn.Module):
    """
    Anatomical register bank that learns from segmentation masks during training.
    
    Key insight: Each register specializes in a specific anatomical structure by:
    1. Learning to predict where that structure appears in images (supervised by masks)
    2. Generating conditioning features based on learned anatomical knowledge
    3. At test time, using learned priors to generate anatomically-informed conditioning
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_organs: int = 12,
        spatial_resolution: int = 8,
        max_timestep: int = 1000,
        organ_names: Optional[List[str]] = None,
        # New: anatomical learning parameters
        anatomical_supervision_weight: float = 1.0,
        anatomical_consistency_weight: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_organs = num_organs
        self.spatial_resolution = spatial_resolution
        self.anatomical_supervision_weight = anatomical_supervision_weight
        self.anatomical_consistency_weight = anatomical_consistency_weight
        
        # Default organ mapping for chest X-rays
        if organ_names is None:
            self.organ_names = [
                "background", "heart", "left_lung", "right_lung", "liver", 
                "left_ribs", "right_ribs", "spine", "clavicle", "diaphragm",
                "mediastinum", "soft_tissue"
            ][:num_organs]
        else:
            self.organ_names = organ_names
        
        # Core components
        self.spatial_pos_embed = SpatialPositionEmbedding(d_model, spatial_resolution, spatial_resolution)
        self.timestep_embed = TimestepAwareEmbedding(d_model, max_timestep)
        
        # Key innovation: Anatomical Structure Learning
        self._build_anatomical_learning_components()
        
        # Conditioning generation (same as before)
        self._build_conditioning_components()
        
        # Initialize with anatomical priors
        self._init_anatomical_priors()
    
    def _build_anatomical_learning_components(self):
        """Build components for learning anatomical structure from masks."""
        
        # 1. Anatomical Structure Predictor
        # This learns to predict anatomical masks from image features
        self.anatomical_predictor = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # 4 = latent channels
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, self.num_organs, 1),  # Predict per-organ masks
        )
        
        # 2. Structure-Aware Register Embeddings
        # Each register learns to represent a specific anatomical structure
        self.structure_embeddings = nn.Parameter(torch.randn(self.num_organs, self.d_model))
        
        # 3. Anatomical Attention Maps
        # Learn where each anatomical structure typically appears
        self.structure_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, self.spatial_resolution * self.spatial_resolution),
                nn.Softmax(dim=-1)
            ) for _ in range(self.num_organs)
        ])
        
        # 4. Anatomical Consistency Enforcer
        # Ensures anatomical structures maintain realistic spatial relationships
        self.anatomical_consistency = nn.Sequential(
            nn.Linear(self.num_organs * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_organs * self.spatial_resolution * self.spatial_resolution),
        )
    
    def _build_conditioning_components(self):
        """Build components for generating conditioning features."""
        
        # Cross-organ interaction
        self.organ_interaction = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.d_model)
        
        # Stage-specific organ weights (learned during training)
        self.stage_organ_weights = nn.ModuleList([
            nn.Linear(self.d_model, self.num_organs) for _ in range(3)  # early, middle, late
        ])
    
    def _init_anatomical_priors(self):
        """Initialize with anatomical priors for chest X-rays."""
        
        with torch.no_grad():
            # Initialize structure embeddings with small random values
            nn.init.xavier_uniform_(self.structure_embeddings, gain=0.1)
            
            # Initialize anatomical attention with rough anatomical priors
            # These are rough estimates that will be refined during training
            anatomical_priors = {
                "heart": (0.3, 0.6),        # Left-center
                "left_lung": (0.2, 0.4),    # Left side
                "right_lung": (0.6, 0.4),   # Right side
                "spine": (0.5, 0.5),        # Center
                "liver": (0.7, 0.8),        # Right-lower
            }
            
            for i, organ_name in enumerate(self.organ_names):
                if organ_name in anatomical_priors:
                    cx, cy = anatomical_priors[organ_name]
                    # Initialize attention to focus on anatomically correct regions
                    attention_map = torch.zeros(self.spatial_resolution, self.spatial_resolution)
                    center_x = int(cx * self.spatial_resolution)
                    center_y = int(cy * self.spatial_resolution)
                    
                    # Gaussian blob around anatomical center
                    for y in range(self.spatial_resolution):
                        for x in range(self.spatial_resolution):
                            dist = ((x - center_x)**2 + (y - center_y)**2) ** 0.5
                            attention_map[y, x] = math.exp(-dist / 2.0)
                    
                    attention_map = attention_map / attention_map.sum()
                    
                    # Set final layer bias to match this prior
                    if len(self.structure_attention[i]) >= 2:
                        self.structure_attention[i][-2].bias.data = attention_map.flatten()
    
    def predict_anatomical_structures(
        self, 
        latent_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict anatomical structure locations from latent features.
        Used during training to learn anatomical representations.
        
        Args:
            latent_features: Latent features from VAE [B, C, H, W]
            
        Returns:
            Dict containing anatomical predictions and attention maps
        """
        batch_size = latent_features.shape[0]
        
        # Predict anatomical masks
        anatomical_logits = self.anatomical_predictor(latent_features)  # [B, num_organs, H, W]
        anatomical_probs = F.softmax(anatomical_logits, dim=1)
        
        # Generate structure-specific attention maps
        structure_attention_maps = []
        for i in range(self.num_organs):
            # Each structure embedding generates its attention pattern
            structure_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)  # [B, d_model]
            attention_map = self.structure_attention[i](structure_embed)  # [B, H*W]
            attention_map = attention_map.view(batch_size, self.spatial_resolution, self.spatial_resolution)
            structure_attention_maps.append(attention_map)
        
        structure_attention_maps = torch.stack(structure_attention_maps, dim=1)  # [B, num_organs, H, W]
        
        return {
            "anatomical_logits": anatomical_logits,
            "anatomical_probs": anatomical_probs,
            "structure_attention_maps": structure_attention_maps,
        }
    
    def compute_anatomical_supervision_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervision loss using ground truth anatomical masks.
        
        Args:
            predictions: Output from predict_anatomical_structures()
            target_masks: Ground truth masks [B, H, W] with class indices
            
        Returns:
            Dict of loss components
        """
        anatomical_logits = predictions["anatomical_logits"]
        structure_attention_maps = predictions["structure_attention_maps"]
        
        # Resize target masks to match prediction resolution
        target_resized = F.interpolate(
            target_masks.float().unsqueeze(1),
            size=anatomical_logits.shape[-2:],
            mode='nearest'
        ).squeeze(1).long()
        
        # 1. Direct anatomical segmentation loss
        segmentation_loss = F.cross_entropy(anatomical_logits, target_resized)
        
        # 2. Structure attention consistency loss
        # Encourage attention maps to align with true anatomical locations
        target_onehot = F.one_hot(target_resized, num_classes=self.num_organs).permute(0, 3, 1, 2).float()
        attention_consistency_loss = F.mse_loss(structure_attention_maps, target_onehot)
        
        # 3. Anatomical spatial relationship loss
        # Encourage realistic spatial relationships between organs
        spatial_relationship_loss = self._compute_spatial_relationship_loss(
            structure_attention_maps, target_onehot
        )
        
        total_loss = (
            segmentation_loss +
            self.anatomical_consistency_weight * attention_consistency_loss +
            0.1 * spatial_relationship_loss
        )
        
        return {
            "total_anatomical_loss": total_loss,
            "segmentation_loss": segmentation_loss,
            "attention_consistency_loss": attention_consistency_loss,
            "spatial_relationship_loss": spatial_relationship_loss,
        }
    
    def _compute_spatial_relationship_loss(
        self,
        predicted_attention: torch.Tensor,
        target_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for anatomical spatial relationships."""
        
        # Encourage heart to be left of center
        if "heart" in self.organ_names:
            heart_idx = self.organ_names.index("heart")
            heart_attention = predicted_attention[:, heart_idx]  # [B, H, W]
            
            # Create coordinate grids
            h, w = heart_attention.shape[-2:]
            y_coords = torch.linspace(-1, 1, h, device=heart_attention.device)
            x_coords = torch.linspace(-1, 1, w, device=heart_attention.device)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
            x_grid = x_grid.T  # [H, W]
            
            # Heart should have negative x center of mass (left side)
            heart_center_x = (heart_attention * x_grid.unsqueeze(0)).sum(dim=(1, 2)) / (heart_attention.sum(dim=(1, 2)) + 1e-6)
            heart_loss = F.relu(heart_center_x + 0.2).mean()  # Penalize if center is too far right
        else:
            heart_loss = 0.0
        
        # Encourage lung symmetry
        lung_loss = 0.0
        if "left_lung" in self.organ_names and "right_lung" in self.organ_names:
            left_idx = self.organ_names.index("left_lung")
            right_idx = self.organ_names.index("right_lung")
            
            left_attention = predicted_attention[:, left_idx]
            right_attention = predicted_attention[:, right_idx]
            
            # Encourage similar attention patterns (but mirrored)
            right_flipped = torch.flip(right_attention, dims=[-1])  # Flip horizontally
            symmetry_loss = F.mse_loss(left_attention, right_flipped)
            lung_loss = symmetry_loss
        
        return heart_loss + lung_loss
    
    def forward(
        self,
        batch_size: int,
        timestep: torch.Tensor,
        device: torch.device,
        latent_height: int = 8,
        latent_width: int = 8,
        latent_features: Optional[torch.Tensor] = None,  # For training with supervision
    ) -> Dict[str, torch.Tensor]:
        """
        Generate anatomical conditioning features.
        
        During training: Uses learned anatomical structure predictions
        During inference: Uses learned anatomical priors
        
        Args:
            batch_size: Batch size
            timestep: Diffusion timestep [B]
            device: Target device
            latent_height: Height of latent space
            latent_width: Width of latent space
            latent_features: Latent features for anatomical prediction (training only)
            
        Returns:
            Dict containing anatomical conditioning features
        """
        
        # Get timestep-aware embeddings
        time_embed_dict = self.timestep_embed(timestep)
        timestep_embed = time_embed_dict["timestep_embed"]  # [B, d_model]
        stage_weights = time_embed_dict["stage_weights"]    # [B, 3]
        
        # Get spatial position embeddings
        spatial_embed = self.spatial_pos_embed(latent_height, latent_width, device)
        spatial_embed = spatial_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Key innovation: Use learned anatomical structure knowledge
        if latent_features is not None and self.training:
            # Training mode: Use anatomical structure predictions
            anatomical_predictions = self.predict_anatomical_structures(latent_features)
            structure_attention_maps = anatomical_predictions["structure_attention_maps"]
        else:
            # Inference mode: Use learned anatomical priors
            structure_attention_maps = []
            for i in range(self.num_organs):
                structure_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)
                attention_map = self.structure_attention[i](structure_embed)
                attention_map = attention_map.view(batch_size, latent_height, latent_width)
                structure_attention_maps.append(attention_map)
            structure_attention_maps = torch.stack(structure_attention_maps, dim=1)
        
        # Generate organ-specific conditioning based on learned structure knowledge
        organ_conditioning = []
        for i in range(self.num_organs):
            # Combine structure embedding with spatial attention
            structure_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)  # [B, d_model]
            attention_map = structure_attention_maps[:, i]  # [B, H, W]
            
            # Weight the embedding by attention (where this organ typically appears)
            attention_weight = attention_map.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            weighted_embed = structure_embed * attention_weight  # [B, d_model]
            
            organ_conditioning.append(weighted_embed)
        
        organ_conditioning = torch.stack(organ_conditioning, dim=1)  # [B, num_organs, d_model]
        
        # Apply stage-aware weighting (same as before)
        stage_modulated_organs = []
        for stage_idx in range(3):
            stage_weight = stage_weights[:, stage_idx:stage_idx+1]  # [B, 1]
            organ_weights = torch.sigmoid(
                self.stage_organ_weights[stage_idx](timestep_embed)
            )  # [B, num_organs]
            
            weighted_organs = organ_conditioning * organ_weights.unsqueeze(-1)
            stage_modulated_organs.append(weighted_organs * stage_weight.unsqueeze(-1))
        
        modulated_organs = sum(stage_modulated_organs)  # [B, num_organs, d_model]
        
        # Add timestep information
        timestep_broadcast = timestep_embed.unsqueeze(1).expand(-1, self.num_organs, -1)
        organ_with_time = modulated_organs + timestep_broadcast * 0.1
        
        # Cross-organ interaction
        interacted_organs = self.organ_interaction(organ_with_time)
        
        # Final projection
        conditioning_features = self.output_proj(interacted_organs)
        
        # Create dense conditioning for cross-attention
        dense_features = []
        for organ_idx in range(self.num_organs):
            organ_feat = conditioning_features[:, organ_idx:organ_idx+1, :].expand(
                -1, latent_height * latent_width, -1
            )
            dense_features.append(organ_feat)
        
        dense_conditioning = torch.stack(dense_features, dim=2)
        dense_conditioning = dense_conditioning.view(
            batch_size, latent_height * latent_width * self.num_organs, self.d_model
        )
        
        result = {
            "organ_conditioning": conditioning_features,
            "spatial_conditioning": spatial_embed,
            "dense_conditioning": dense_conditioning,
            "timestep_embed": timestep_embed,
            "stage_weights": stage_weights,
            "organ_names": self.organ_names,
            "structure_attention_maps": structure_attention_maps,
        }
        
        # Include anatomical predictions during training
        if latent_features is not None and self.training:
            result["anatomical_predictions"] = anatomical_predictions
        
        return result


def create_supervised_anatomical_registers(
    latent_resolution: int = 8,
    d_model: int = 512,
    num_organs: int = 12,
    **kwargs
) -> SupervisedAnatomicalRegisterBank:
    """
    Create supervised anatomical register bank.
    
    Args:
        latent_resolution: Resolution of latent space
        d_model: Model dimension
        num_organs: Number of anatomical organs
        **kwargs: Additional arguments
        
    Returns:
        Configured SupervisedAnatomicalRegisterBank
    """
    return SupervisedAnatomicalRegisterBank(
        d_model=d_model,
        num_organs=num_organs,
        spatial_resolution=latent_resolution,
        **kwargs
    )