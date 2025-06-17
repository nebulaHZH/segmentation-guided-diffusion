"""
Example usage of general supervised registers for CT-org dataset.
Demonstrates how to use the code with any multiclass medical dataset.
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from .general_supervised_registers import create_general_supervised_registers
from .anatomical_unet import AnatomicalUNet2DConditionModel
from .train_supervised_ldm import SupervisedLDMDataset, SupervisedLDMTrainer
from .vae import AnatomicalVAE

# CT-org dataset class mapping
CT_ORG_CLASSES = [
    "background",     # 0
    "liver",         # 1  
    "bladder",       # 2
    "lungs",         # 3
    "kidneys",       # 4
    "bone",          # 5
    "brain",         # 6
    "heart",         # 7
    "pancreas",      # 8
    "spleen",        # 9
    "gallbladder",   # 10
    "esophagus",     # 11
    "stomach",       # 12
    "aorta",         # 13
]

def setup_ct_org_training(
    train_image_dir: str,
    train_mask_dir: str,
    vae_path: str,
    output_dir: str = "outputs/ct_org_ldm",
    device: str = "cuda",
):
    """
    Set up training for CT-org dataset.
    
    Args:
        train_image_dir: Directory with CT images
        train_mask_dir: Directory with corresponding segmentation masks
        vae_path: Path to pretrained VAE
        output_dir: Output directory for training
        device: Device to use
    """
    
    # 1. Load pretrained VAE
    print("Loading VAE...")
    vae = AnatomicalVAE.from_pretrained(vae_path)
    
    # 2. Create dataset
    print("Creating CT-org dataset...")
    train_dataset = SupervisedLDMDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        image_size=512,
        extensions=('.png', '.jpg', '.jpeg', '.dcm'),  # Support DICOM
        mask_extensions=('.png', '.npy'),
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,  # Smaller batch for 3D CT data
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # 3. Create UNet with general supervised registers
    print("Creating UNet with CT-org anatomical registers...")
    unet = AnatomicalUNet2DConditionModel(
        sample_size=64,  # 512 // 8
        in_channels=4,
        out_channels=4,
        cross_attention_dim=768,
        anatomical_conditioning_dim=512,
        anatomical_num_organs=len(CT_ORG_CLASSES),
    )
    
    # 4. Replace register bank with general supervised version
    unet.anatomical_registers = create_general_supervised_registers(
        num_classes=len(CT_ORG_CLASSES),
        class_names=CT_ORG_CLASSES,
        latent_resolution=8,
        d_model=512,
        enable_spatial_relationships=True,
        spatial_smoothness_weight=0.1,
    )
    
    # 5. Create trainer
    print("Setting up trainer...")
    trainer = SupervisedLDMTrainer(
        unet=unet,
        vae=vae,
        noise_scheduler=torch.hub.load('huggingface/pytorch-transformers', 'ddpmscheduler'),
        train_dataloader=train_dataloader,
        device=device,
        output_dir=output_dir,
        learning_rate=1e-4,
        anatomical_supervision_weight=1.0,  # Strong supervision initially
        supervision_probability=0.8,  # Use supervision 80% of the time
        use_anatomical_supervision=True,
        use_ema=True,
        use_wandb=True,
        wandb_project="ct-org-anatomical-ldm",
    )
    
    return trainer

def train_ct_org_model():
    """Example training script for CT-org."""
    
    # Paths (adjust these for your data)
    train_image_dir = "/path/to/ct_org/images/train"
    train_mask_dir = "/path/to/ct_org/labels/train"
    vae_path = "/path/to/pretrained/vae"
    
    # Setup training
    trainer = setup_ct_org_training(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        vae_path=vae_path,
        output_dir="outputs/ct_org_ldm",
    )
    
    # Progressive training strategy
    print("Stage 1: Strong anatomical supervision...")
    trainer.anatomical_supervision_weight = 2.0
    trainer.supervision_probability = 1.0
    trainer.train(num_epochs=100, save_every=20)
    
    print("Stage 2: Balanced supervision...")
    trainer.anatomical_supervision_weight = 0.5
    trainer.supervision_probability = 0.6
    trainer.train(num_epochs=200, save_every=20)
    
    print("Stage 3: Fine-tuning with minimal supervision...")
    trainer.anatomical_supervision_weight = 0.1
    trainer.supervision_probability = 0.3
    trainer.train(num_epochs=100, save_every=20)
    
    print("Training complete!")

def setup_brain_mri_training():
    """Example for brain MRI dataset with different organs."""
    
    BRAIN_MRI_CLASSES = [
        "background",
        "cerebrospinal_fluid",
        "gray_matter", 
        "white_matter",
        "ventricles",
        "brainstem",
        "cerebellum",
        "skull",
    ]
    
    # Same setup but with different class names
    registers = create_general_supervised_registers(
        num_classes=len(BRAIN_MRI_CLASSES),
        class_names=BRAIN_MRI_CLASSES,
        latent_resolution=8,
        d_model=512,
    )
    
    print(f"Created brain MRI registers for {len(BRAIN_MRI_CLASSES)} classes:")
    for i, name in enumerate(BRAIN_MRI_CLASSES):
        print(f"  Class {i}: {name}")
    
    return registers

def setup_cardiac_ct_training():
    """Example for cardiac CT dataset."""
    
    CARDIAC_CT_CLASSES = [
        "background",
        "myocardium",
        "left_ventricle",
        "right_ventricle", 
        "left_atrium",
        "right_atrium",
        "aorta",
        "pulmonary_artery",
        "coronary_arteries",
        "pericardium",
    ]
    
    registers = create_general_supervised_registers(
        num_classes=len(CARDIAC_CT_CLASSES),
        class_names=CARDIAC_CT_CLASSES,
        latent_resolution=8,
        d_model=512,
        enable_spatial_relationships=True,  # Important for cardiac anatomy
    )
    
    return registers

if __name__ == "__main__":
    # Example usage
    print("CT-org example:")
    train_ct_org_model()
    
    print("\nBrain MRI example:")
    brain_registers = setup_brain_mri_training()
    
    print("\nCardiac CT example:")
    cardiac_registers = setup_cardiac_ct_training()
    
    print("\nGeneral supervised registers work with any multiclass medical dataset!")
    print("Key requirements:")
    print("1. Images paired with segmentation masks")
    print("2. Masks contain class indices (0, 1, 2, ...)")
    print("3. Provide class names for interpretability")
    print("4. Each class gets one dedicated register")