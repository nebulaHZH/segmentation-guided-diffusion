#!/usr/bin/env python3
"""
Complete training script for your custom dataset with multiclass masks.
Modify the YOUR_CLASSES list and dataset paths for your specific dataset.
"""

import torch
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from .train_supervised_ldm import SupervisedLDMDataset, SupervisedLDMTrainer
from .general_supervised_registers import create_general_supervised_registers
from .anatomical_unet import AnatomicalUNet2DConditionModel
from .vae import AnatomicalVAE
from diffusers import DDPMScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# MODIFY THIS FOR YOUR DATASET
# ========================================

# Define your dataset classes (modify this list!)
YOUR_CLASSES = [
    "background",        # Class 0 (always background)
    "liver",            # Class 1
    "right_kidney",     # Class 2
    "left_kidney",      # Class 3
    "spleen",           # Class 4
    "pancreas",         # Class 5
    "gallbladder",      # Class 6
    "stomach",          # Class 7
    "aorta",            # Class 8
    "inferior_vena_cava", # Class 9
    "muscle",           # Class 10
    "bone",             # Class 11
    # Add/remove classes as needed for your dataset
]

NUM_CLASSES = len(YOUR_CLASSES)

def create_datasets(args):
    """Create training and validation datasets."""
    
    logger.info(f"Creating datasets for {NUM_CLASSES} classes: {YOUR_CLASSES}")
    
    # Training dataset
    train_dataset = SupervisedLDMDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=args.image_size,
        extensions=('.png', '.jpg', '.jpeg', '.dcm'),
        mask_extensions=('.png', '.npy'),
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Validation dataset (optional)
    val_dataloader = None
    if args.val_images and args.val_masks:
        val_dataset = SupervisedLDMDataset(
            image_dir=args.val_images,
            mask_dir=args.val_masks,
            image_size=args.image_size,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    logger.info(f"Created training dataset with {len(train_dataset)} samples")
    return train_dataloader, val_dataloader

def create_model(args):
    """Create UNet with supervised anatomical registers."""
    
    logger.info("Creating UNet with supervised anatomical registers...")
    
    # Create UNet
    unet = AnatomicalUNet2DConditionModel(
        sample_size=args.image_size // 8,  # Latent resolution
        in_channels=4,  # VAE latent channels
        out_channels=4,
        cross_attention_dim=768,
        anatomical_conditioning_dim=args.anatomical_dim,
        anatomical_num_organs=NUM_CLASSES,
    )
    
    # Replace with general supervised registers
    unet.anatomical_registers = create_general_supervised_registers(
        num_classes=NUM_CLASSES,
        class_names=YOUR_CLASSES,
        latent_resolution=args.image_size // 64,  # 8x8 for 512px images
        d_model=args.anatomical_dim,
        enable_spatial_relationships=True,
        spatial_smoothness_weight=0.1,
    )
    
    logger.info(f"Created UNet with {NUM_CLASSES} anatomical registers")
    logger.info(f"Register mapping: {list(enumerate(YOUR_CLASSES))}")
    
    return unet

def progressive_training(trainer, args):
    """Execute progressive training strategy."""
    
    logger.info("Starting progressive training strategy...")
    
    # Stage 1: Strong anatomical supervision
    logger.info("=" * 60)
    logger.info("STAGE 1: STRONG ANATOMICAL SUPERVISION")
    logger.info("=" * 60)
    trainer.anatomical_supervision_weight = args.stage1_weight
    trainer.supervision_probability = args.stage1_prob
    
    logger.info(f"Supervision weight: {trainer.anatomical_supervision_weight}")
    logger.info(f"Supervision probability: {trainer.supervision_probability}")
    
    trainer.train(
        num_epochs=args.stage1_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    # Stage 2: Balanced supervision
    logger.info("=" * 60)
    logger.info("STAGE 2: BALANCED SUPERVISION")
    logger.info("=" * 60)
    trainer.anatomical_supervision_weight = args.stage2_weight
    trainer.supervision_probability = args.stage2_prob
    
    logger.info(f"Supervision weight: {trainer.anatomical_supervision_weight}")
    logger.info(f"Supervision probability: {trainer.supervision_probability}")
    
    trainer.train(
        num_epochs=args.stage2_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    # Stage 3: Fine-tuning with minimal supervision
    logger.info("=" * 60)
    logger.info("STAGE 3: FINE-TUNING WITH MINIMAL SUPERVISION")
    logger.info("=" * 60)
    trainer.anatomical_supervision_weight = args.stage3_weight
    trainer.supervision_probability = args.stage3_prob
    
    logger.info(f"Supervision weight: {trainer.anatomical_supervision_weight}")
    logger.info(f"Supervision probability: {trainer.supervision_probability}")
    
    trainer.train(
        num_epochs=args.stage3_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every
    )
    
    logger.info("Progressive training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train Anatomical LDM on your dataset")
    
    # Dataset arguments
    parser.add_argument("--train_images", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--train_masks", type=str, required=True,
                        help="Directory containing training masks")
    parser.add_argument("--val_images", type=str, default=None,
                        help="Directory containing validation images")
    parser.add_argument("--val_masks", type=str, default=None,
                        help="Directory containing validation masks")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to pretrained VAE")
    
    # Model arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--anatomical_dim", type=int, default=512,
                        help="Anatomical conditioning dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (adjust for your GPU)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    
    # Progressive training stages
    parser.add_argument("--stage1_epochs", type=int, default=100,
                        help="Epochs for stage 1 (strong supervision)")
    parser.add_argument("--stage1_weight", type=float, default=2.0,
                        help="Anatomical supervision weight for stage 1")
    parser.add_argument("--stage1_prob", type=float, default=1.0,
                        help="Supervision probability for stage 1")
    
    parser.add_argument("--stage2_epochs", type=int, default=200,
                        help="Epochs for stage 2 (balanced)")
    parser.add_argument("--stage2_weight", type=float, default=0.5,
                        help="Anatomical supervision weight for stage 2")
    parser.add_argument("--stage2_prob", type=float, default=0.6,
                        help="Supervision probability for stage 2")
    
    parser.add_argument("--stage3_epochs", type=int, default=100,
                        help="Epochs for stage 3 (fine-tuning)")
    parser.add_argument("--stage3_weight", type=float, default=0.1,
                        help="Anatomical supervision weight for stage 3")
    parser.add_argument("--stage3_prob", type=float, default=0.3,
                        help="Supervision probability for stage 3")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/anatomical_ldm",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=20,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--validate_every", type=int, default=10,
                        help="Validate every N epochs")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="anatomical-ldm",
                        help="W&B project name")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA for model weights")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.train_images).exists():
        raise ValueError(f"Training images directory does not exist: {args.train_images}")
    if not Path(args.train_masks).exists():
        raise ValueError(f"Training masks directory does not exist: {args.train_masks}")
    if not Path(args.vae_path).exists():
        raise ValueError(f"VAE path does not exist: {args.vae_path}")
    
    logger.info("=" * 80)
    logger.info("ANATOMICAL LDM TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset classes ({NUM_CLASSES}): {YOUR_CLASSES}")
    logger.info(f"Training images: {args.train_images}")
    logger.info(f"Training masks: {args.train_masks}")
    logger.info(f"VAE path: {args.vae_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load VAE
    logger.info("Loading pretrained VAE...")
    vae = AnatomicalVAE.from_pretrained(args.vae_path)
    logger.info("VAE loaded successfully")
    
    # Create datasets
    train_dataloader, val_dataloader = create_datasets(args)
    
    # Create model
    unet = create_model(args)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = SupervisedLDMTrainer(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        anatomical_supervision_weight=args.stage1_weight,  # Will be updated in progressive training
        supervision_probability=args.stage1_prob,
        use_anatomical_supervision=True,
        use_ema=args.use_ema,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming training from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Execute progressive training
    progressive_training(trainer, args)
    
    # Final save
    final_path = Path(args.output_dir) / "final_model"
    trainer.save_pipeline(final_path)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Final model saved to: {final_path}")
    logger.info(f"Total classes trained: {NUM_CLASSES}")
    logger.info("You can now use this model for generation!")

if __name__ == "__main__":
    main()