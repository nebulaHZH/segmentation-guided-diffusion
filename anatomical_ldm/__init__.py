"""
Anatomical Latent Diffusion Models for Medical Imaging.

This package implements a novel approach to diffusion models that incorporates
anatomical knowledge through cross-attention conditioning in latent space.

Key components:
- AnatomicalVAE: VAE with anatomical consistency features
- AnatomicalRegisterBank: Organ-specific conditioning system
- AnatomicalUNet2DConditionModel: UNet with anatomical cross-attention
- Training pipelines for both VAE and LDM
- Comprehensive evaluation utilities
"""

from .vae import AnatomicalVAE, create_anatomical_vae
from .anatomical_registers import AnatomicalRegisterBank, create_anatomical_registers
from .anatomical_unet import AnatomicalUNet2DConditionModel, create_anatomical_unet
from .train_ldm import AnatomicalLDMPipeline
from .general_supervised_registers import GeneralSupervisedAnatomicalRegisterBank, create_general_supervised_registers

__version__ = "0.2.0"
__author__ = "Anatomical LDM Research Team"

__all__ = [
    "AnatomicalVAE",
    "create_anatomical_vae", 
    "AnatomicalRegisterBank",
    "create_anatomical_registers",
    "AnatomicalUNet2DConditionModel",
    "create_anatomical_unet",
    "AnatomicalLDMPipeline",
    "GeneralSupervisedAnatomicalRegisterBank",
    "create_general_supervised_registers",
]