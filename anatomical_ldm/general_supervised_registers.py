"""
General Supervised Anatomical Register Bank that works with any multiclass medical dataset.
Fixed version that removes chest X-ray specific assumptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .anatomical_registers import SpatialPositionEmbedding, TimestepAwareEmbedding


class GeneralSupervisedAnatomicalRegisterBank(nn.Module):
    """
    General anatomical register bank that learns from any multiclass segmentation masks.
    
    Works with any medical imaging modality and organ set without hardcoded assumptions.
    Each mask class index corresponds to one anatomical register.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 12,  # Number of classes in segmentation masks
        spatial_resolution: int = 8,
        max_timestep: int = 1000,
        class_names: Optional[List[str]] = None,
        # Anatomical learning parameters
        anatomical_supervision_weight: float = 1.0,
        anatomical_consistency_weight: float = 0.5,
        # General spatial relationship learning
        enable_spatial_relationships: bool = True,
        spatial_smoothness_weight: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.spatial_resolution = spatial_resolution
        self.anatomical_supervision_weight = anatomical_supervision_weight
        self.anatomical_consistency_weight = anatomical_consistency_weight
        self.enable_spatial_relationships = enable_spatial_relationships
        self.spatial_smoothness_weight = spatial_smoothness_weight
        
        # Generic class names (can be overridden)
        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            assert len(class_names) == num_classes, f"Must provide {num_classes} class names"
            self.class_names = class_names
        
        # Core components
        self.spatial_pos_embed = SpatialPositionEmbedding(d_model, spatial_resolution, spatial_resolution)
        self.timestep_embed = TimestepAwareEmbedding(d_model, max_timestep)
        
        # Build anatomical learning components
        self._build_anatomical_learning_components()
        
        # Build conditioning components
        self._build_conditioning_components()
        
        # Initialize without hardcoded priors
        self._init_parameters()
    
    def _build_anatomical_learning_components(self):
        """Build components for learning anatomical structure from masks."""
        
        # 1. Anatomical Structure Predictor
        # Predicts class masks from latent features
        self.anatomical_predictor = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),  # 4 = latent channels
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, self.num_classes, 1),  # Predict per-class masks
        )
        
        # 2. Structure-Aware Register Embeddings
        # Each register learns to represent a specific anatomical class
        self.structure_embeddings = nn.Parameter(torch.randn(self.num_classes, self.d_model))
        
        # 3. Anatomical Attention Maps
        # Learn where each anatomical class typically appears
        self.structure_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.LayerNorm(self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model // 2, self.spatial_resolution * self.spatial_resolution),
                nn.Softmax(dim=-1)
            ) for _ in range(self.num_classes)
        ])
        
        # 4. General Spatial Consistency Enforcer
        # Learns spatial relationships between classes without hardcoded assumptions
        if self.enable_spatial_relationships:
            self.spatial_consistency = nn.Sequential(
                nn.Linear(self.num_classes * self.d_model, self.d_model * 2),
                nn.LayerNorm(self.d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.num_classes * self.spatial_resolution * self.spatial_resolution),
            )
    
    def _build_conditioning_components(self):
        """Build components for generating conditioning features."""
        
        # Cross-class interaction
        self.class_interaction = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        
        # Stage-specific class weights (learned during training)
        self.stage_class_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, self.num_classes),
                nn.Sigmoid()
            ) for _ in range(3)  # early, middle, late
        ])
    
    def _init_parameters(self):
        """Initialize parameters without modality-specific assumptions."""
        
        with torch.no_grad():
            # Initialize structure embeddings with Xavier uniform
            nn.init.xavier_uniform_(self.structure_embeddings, gain=0.1)
            
            # Initialize attention networks to output uniform distributions initially
            for attention_net in self.structure_attention:
                # Set final layer bias to produce uniform attention
                uniform_bias = torch.log(torch.ones(self.spatial_resolution * self.spatial_resolution))
                uniform_bias = uniform_bias - uniform_bias.logsumexp(dim=0)  # Normalize for softmax
                attention_net[-2].bias.data = uniform_bias
            
            # Initialize stage weights to be neutral initially
            for stage_net in self.stage_class_weights:
                # Initialize to output 0.5 for all classes (neutral weighting)
                stage_net[-2].bias.data.fill_(0.0)  # Sigmoid(0) = 0.5
    
    def predict_anatomical_structures(
        self, 
        latent_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict anatomical class locations from latent features.
        
        Args:
            latent_features: Latent features from VAE [B, C, H, W]
            
        Returns:
            Dict containing anatomical predictions and attention maps
        """
        batch_size = latent_features.shape[0]
        
        # Predict class masks
        anatomical_logits = self.anatomical_predictor(latent_features)  # [B, num_classes, H, W]
        anatomical_probs = F.softmax(anatomical_logits, dim=1)
        
        # Generate class-specific attention maps
        structure_attention_maps = []
        for i in range(self.num_classes):
            # Each class embedding generates its attention pattern
            class_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)  # [B, d_model]
            attention_map = self.structure_attention[i](class_embed)  # [B, H*W]
            attention_map = attention_map.view(batch_size, self.spatial_resolution, self.spatial_resolution)
            structure_attention_maps.append(attention_map)
        
        structure_attention_maps = torch.stack(structure_attention_maps, dim=1)  # [B, num_classes, H, W]
        
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
        Compute supervision loss using ground truth masks.
        
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
        
        # 1. Direct segmentation loss
        segmentation_loss = F.cross_entropy(anatomical_logits, target_resized, ignore_index=-1)
        
        # 2. Attention consistency loss
        # Encourage attention maps to align with true class locations
        target_onehot = F.one_hot(
            target_resized.clamp(min=0), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        
        attention_consistency_loss = F.mse_loss(structure_attention_maps, target_onehot)
        
        # 3. General spatial relationship learning
        spatial_relationship_loss = self._compute_general_spatial_loss(
            structure_attention_maps, target_onehot
        )
        
        # 4. Spatial smoothness loss (encourage coherent regions)
        smoothness_loss = self._compute_spatial_smoothness_loss(structure_attention_maps)
        
        total_loss = (
            segmentation_loss +
            self.anatomical_consistency_weight * attention_consistency_loss +
            0.1 * spatial_relationship_loss +
            self.spatial_smoothness_weight * smoothness_loss
        )
        
        return {
            "total_anatomical_loss": total_loss,
            "segmentation_loss": segmentation_loss,
            "attention_consistency_loss": attention_consistency_loss,
            "spatial_relationship_loss": spatial_relationship_loss,
            "smoothness_loss": smoothness_loss,
        }
    
    def _compute_general_spatial_loss(
        self,
        predicted_attention: torch.Tensor,
        target_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute general spatial relationship loss without modality-specific assumptions.
        Encourages learning of spatial co-occurrence patterns from data.
        """
        if not self.enable_spatial_relationships:
            return torch.tensor(0.0, device=predicted_attention.device)
        
        batch_size, num_classes, h, w = predicted_attention.shape
        
        # Compute center of mass for each class
        y_coords = torch.linspace(-1, 1, h, device=predicted_attention.device)
        x_coords = torch.linspace(-1, 1, w, device=predicted_attention.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Centers of mass for predicted and target
        pred_centers = []
        target_centers = []
        
        for class_idx in range(num_classes):
            pred_attention = predicted_attention[:, class_idx]  # [B, H, W]
            target_attention_class = target_attention[:, class_idx]  # [B, H, W]
            
            # Predicted center of mass
            pred_mass = pred_attention.sum(dim=(1, 2), keepdim=True) + 1e-6
            pred_center_y = (pred_attention * y_grid.unsqueeze(0)).sum(dim=(1, 2)) / pred_mass.squeeze()
            pred_center_x = (pred_attention * x_grid.unsqueeze(0)).sum(dim=(1, 2)) / pred_mass.squeeze()
            pred_centers.append(torch.stack([pred_center_y, pred_center_x], dim=1))
            
            # Target center of mass
            target_mass = target_attention_class.sum(dim=(1, 2), keepdim=True) + 1e-6
            target_center_y = (target_attention_class * y_grid.unsqueeze(0)).sum(dim=(1, 2)) / target_mass.squeeze()
            target_center_x = (target_attention_class * x_grid.unsqueeze(0)).sum(dim=(1, 2)) / target_mass.squeeze()
            target_centers.append(torch.stack([target_center_y, target_center_x], dim=1))
        
        pred_centers = torch.stack(pred_centers, dim=1)  # [B, num_classes, 2]
        target_centers = torch.stack(target_centers, dim=1)  # [B, num_classes, 2]
        
        # Encourage predicted centers to match target centers
        center_loss = F.mse_loss(pred_centers, target_centers)
        
        return center_loss
    
    def _compute_spatial_smoothness_loss(self, attention_maps: torch.Tensor) -> torch.Tensor:
        """Encourage spatial smoothness in attention maps."""
        
        # Spatial gradients
        grad_h = torch.abs(attention_maps[:, :, 1:, :] - attention_maps[:, :, :-1, :])
        grad_w = torch.abs(attention_maps[:, :, :, 1:] - attention_maps[:, :, :, :-1])
        
        smoothness_loss = grad_h.mean() + grad_w.mean()
        return smoothness_loss
    
    def forward(
        self,
        batch_size: int,
        timestep: torch.Tensor,
        device: torch.device,
        latent_height: int = 8,
        latent_width: int = 8,
        latent_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate anatomical conditioning features.
        
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
        
        # Use learned anatomical structure knowledge
        if latent_features is not None and self.training:
            # Training mode: Use anatomical structure predictions
            anatomical_predictions = self.predict_anatomical_structures(latent_features)
            structure_attention_maps = anatomical_predictions["structure_attention_maps"]
        else:
            # Inference mode: Use learned priors
            structure_attention_maps = []
            for i in range(self.num_classes):
                class_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)
                attention_map = self.structure_attention[i](class_embed)
                attention_map = attention_map.view(batch_size, latent_height, latent_width)
                structure_attention_maps.append(attention_map)
            structure_attention_maps = torch.stack(structure_attention_maps, dim=1)
        
        # Generate class-specific conditioning
        class_conditioning = []
        for i in range(self.num_classes):
            # Combine class embedding with spatial attention
            class_embed = self.structure_embeddings[i:i+1].expand(batch_size, -1)  # [B, d_model]
            attention_map = structure_attention_maps[:, i]  # [B, H, W]
            
            # Weight the embedding by attention (where this class typically appears)
            attention_weight = attention_map.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            weighted_embed = class_embed * (attention_weight + 0.1)  # Prevent zero weighting
            
            class_conditioning.append(weighted_embed)
        
        class_conditioning = torch.stack(class_conditioning, dim=1)  # [B, num_classes, d_model]
        
        # Apply stage-aware weighting
        stage_modulated_classes = []
        for stage_idx in range(3):
            stage_weight = stage_weights[:, stage_idx:stage_idx+1]  # [B, 1]
            class_weights = self.stage_class_weights[stage_idx](timestep_embed)  # [B, num_classes]
            
            weighted_classes = class_conditioning * class_weights.unsqueeze(-1)
            stage_modulated_classes.append(weighted_classes * stage_weight.unsqueeze(-1))
        
        modulated_classes = sum(stage_modulated_classes)  # [B, num_classes, d_model]
        
        # Add timestep information
        timestep_broadcast = timestep_embed.unsqueeze(1).expand(-1, self.num_classes, -1)
        class_with_time = modulated_classes + timestep_broadcast * 0.1
        
        # Cross-class interaction
        interacted_classes = self.class_interaction(class_with_time)
        
        # Final projection
        conditioning_features = self.output_proj(interacted_classes)
        
        # Create dense conditioning for cross-attention
        dense_features = []
        for class_idx in range(self.num_classes):
            class_feat = conditioning_features[:, class_idx:class_idx+1, :].expand(
                -1, latent_height * latent_width, -1
            )
            dense_features.append(class_feat)
        
        dense_conditioning = torch.stack(dense_features, dim=2)
        dense_conditioning = dense_conditioning.view(
            batch_size, latent_height * latent_width * self.num_classes, self.d_model
        )
        
        result = {
            "class_conditioning": conditioning_features,
            "spatial_conditioning": spatial_embed,
            "dense_conditioning": dense_conditioning,
            "timestep_embed": timestep_embed,
            "stage_weights": stage_weights,
            "class_names": self.class_names,
            "structure_attention_maps": structure_attention_maps,
        }
        
        # Include anatomical predictions during training
        if latent_features is not None and self.training:
            result["anatomical_predictions"] = anatomical_predictions
        
        return result


def create_general_supervised_registers(
    num_classes: int,
    class_names: Optional[List[str]] = None,
    latent_resolution: int = 8,
    d_model: int = 512,
    **kwargs
) -> GeneralSupervisedAnatomicalRegisterBank:
    """
    Create general supervised register bank for any multiclass dataset.
    
    Args:
        num_classes: Number of classes in segmentation masks
        class_names: Optional list of class names
        latent_resolution: Resolution of latent space
        d_model: Model dimension
        **kwargs: Additional arguments
        
    Returns:
        Configured GeneralSupervisedAnatomicalRegisterBank
    """
    return GeneralSupervisedAnatomicalRegisterBank(
        d_model=d_model,
        num_classes=num_classes,
        class_names=class_names,
        spatial_resolution=latent_resolution,
        **kwargs
    )