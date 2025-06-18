"""
Training script for Anatomical VAE.
Supports both supervised (with masks) and unsupervised training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb

from .vae import AnatomicalVAE, create_anatomical_vae

logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray images with optional anatomical masks.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        image_size: int = 512,
        extensions: Tuple[str] = ('.png', '.jpg', '.jpeg'),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.glob(f"**/*{ext}")))
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Check for corresponding masks
        self.mask_paths = []
        if self.mask_dir:
            for image_path in self.image_paths:
                # Look for corresponding mask
                relative_path = image_path.relative_to(self.image_dir)
                mask_path = self.mask_dir / relative_path.with_suffix('.png')
                self.mask_paths.append(mask_path if mask_path.exists() else None)
            
            valid_masks = sum(1 for p in self.mask_paths if p is not None)
            logger.info(f"Found {valid_masks} corresponding masks")
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
        
        # Mask transforms (if using masks)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        result = {'image': image}
        
        # Load mask if available
        if self.mask_dir and idx < len(self.mask_paths) and self.mask_paths[idx]:
            mask_path = self.mask_paths[idx]
            try:
                mask = Image.open(mask_path)
                mask = self.mask_transform(mask)
                
                # Convert to multi-class mask if needed
                if mask.shape[0] == 1:
                    # Single channel mask - assume class indices
                    mask = mask.squeeze(0).long()
                    # Convert to one-hot encoding
                    num_classes = mask.max().item() + 1
                    mask_onehot = torch.zeros(num_classes, *mask.shape)
                    mask_onehot.scatter_(0, mask.unsqueeze(0), 1)
                    mask = mask_onehot
                
                result['mask'] = mask
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e}")
        
        return result


class VAETrainer:
    """
    Trainer for Anatomical VAE.
    """
    
    def __init__(
        self,
        vae: AnatomicalVAE,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs/vae',
        learning_rate: float = 1e-4,
        beta_start: float = 1e-6,
        beta_end: float = 1e-2,
        beta_warmup_steps: int = 1000,
        anatomical_loss_weight: float = 0.1,
        gradient_clip: float = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "anatomical-ldm",
    ):
        self.vae = vae.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # KL beta scheduling
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        
        # Loss weights
        self.anatomical_loss_weight = anatomical_loss_weight
        self.gradient_clip = gradient_clip
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, name="vae-training")
            wandb.config.update({
                "learning_rate": learning_rate,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "anatomical_loss_weight": anatomical_loss_weight,
            })
        
        self.global_step = 0
        self.epoch = 0
        
    def get_beta(self, step: int) -> float:
        """Get current KL beta value with warmup."""
        if step < self.beta_warmup_steps:
            alpha = step / self.beta_warmup_steps
            return self.beta_start + alpha * (self.beta_end - self.beta_start)
        else:
            return self.beta_end
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.vae.train()
        
        images = batch['image'].to(self.device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.to(self.device)
        
        # Get current beta
        beta = self.get_beta(self.global_step)
        
        # Forward pass and compute losses
        loss_dict = self.vae.compute_loss(
            sample=images,
            target_masks=masks,
            beta=beta,
        )
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Convert to float for logging
        loss_values = {k: v.item() for k, v in loss_dict.items()}
        loss_values['beta'] = beta
        
        return loss_values
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_dataloader is None:
            return {}
        
        self.vae.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch['image'].to(self.device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(self.device)
                
                # Compute validation losses
                loss_dict = self.vae.compute_loss(
                    sample=images,
                    target_masks=masks,
                    beta=self.beta_end,  # Use final beta for validation
                )
                
                # Accumulate losses
                for k, v in loss_dict.items():
                    if k not in total_losses:
                        total_losses[k] = 0
                    total_losses[k] += v.item()
                
                num_batches += 1
        
        # Average losses
        avg_losses = {f"val_{k}": v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, save_path: Optional[str] = None):
        """Save model checkpoint."""
        if save_path is None:
            save_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.vae.config,
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        # Also save the VAE in diffusers format
        vae_path = self.output_dir / f"vae_epoch_{epoch}"
        self.vae.save_pretrained(vae_path)
        logger.info(f"Saved VAE in diffusers format to {vae_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate sample reconstructions for visualization."""
        self.vae.eval()
        
        with torch.no_grad():
            # Get some real images
            batch = next(iter(self.val_dataloader or self.train_dataloader))
            real_images = batch['image'][:num_samples].to(self.device)
            
            # Encode and decode
            output = self.vae(real_images, sample_posterior=True, return_dict=True)
            reconstructions = output.sample  # Fixed: use .sample instead of ['sample']
            
            # Concatenate real and reconstructed for comparison
            # Top row: original images, Bottom row: reconstructions
            comparison = torch.cat([real_images, reconstructions], dim=0)
            
            return comparison
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 5):
        """Main training loop."""
        logger.info(f"Starting VAE training for {num_epochs} epochs")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training loop
            epoch_losses = {}
            num_batches = 0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False
            )
            
            for batch in progress_bar:
                # Training step
                step_losses = self.train_step(batch)
                
                # Accumulate losses
                for k, v in step_losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0
                    epoch_losses[k] += v
                
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_losses['total_loss']:.4f}",
                    'recon': f"{step_losses['reconstruction_loss']:.4f}",
                    'kl': f"{step_losses['kl_loss']:.4f}",
                    'anat': f"{step_losses['anatomical_loss']:.4f}",
                })
                
                # Log to wandb
                if self.use_wandb and self.global_step % 100 == 0:
                    wandb.log(step_losses, step=self.global_step)
            
            # Average epoch losses
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                val_losses = self.validate()
                avg_losses.update(val_losses)
                
                # Generate reconstruction samples
                if self.val_dataloader or self.train_dataloader:
                    try:
                        samples = self.generate_samples(num_samples=8)  # 8 original + 8 reconstructed = 16 total
                        sample_path = self.output_dir / f"reconstructions_epoch_{epoch+1}.png"
                        save_image(
                            samples, 
                            sample_path, 
                            nrow=8,  # 8 images per row (top row: originals, bottom row: reconstructions)
                            normalize=True, 
                            value_range=(-1, 1),
                            pad_value=1.0  # White padding between images
                        )
                        
                        logger.info(f"Saved reconstruction samples to {sample_path}")
                        
                        if self.use_wandb:
                            wandb.log({
                                "reconstructions": wandb.Image(str(sample_path), 
                                    caption=f"Top row: Original images, Bottom row: VAE reconstructions (Epoch {epoch+1})")
                            }, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to generate reconstruction samples: {e}")
            
            # Logging
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
            logger.info(f"Epoch {epoch+1}: {loss_str}")
            
            if self.use_wandb:
                wandb.log(avg_losses, step=self.global_step)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final checkpoint
        self.save_checkpoint(num_epochs)
        logger.info("Training completed!")


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    val_mask_dir: Optional[str] = None,
    image_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders."""
    
    # Training dataset
    train_dataset = ChestXrayDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        image_size=image_size,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Validation dataset
    val_dataloader = None
    if val_dir:
        val_dataset = ChestXrayDataset(
            image_dir=val_dir,
            mask_dir=val_mask_dir,
            image_size=image_size,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train Anatomical VAE")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Directory containing validation images")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory containing anatomical masks (optional)")
    parser.add_argument("--val_mask_dir", type=str, default=None,
                        help="Directory containing validation masks (optional)")
    
    # Model arguments
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels")
    parser.add_argument("--num_anatomical_regions", type=int, default=12,
                        help="Number of anatomical regions")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--beta_start", type=float, default=1e-6,
                        help="Initial KL beta")
    parser.add_argument("--beta_end", type=float, default=1e-2,
                        help="Final KL beta")
    parser.add_argument("--beta_warmup_steps", type=int, default=1000,
                        help="KL beta warmup steps")
    parser.add_argument("--anatomical_loss_weight", type=float, default=0.1,
                        help="Weight for anatomical consistency loss")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/vae",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--validate_every", type=int, default=5,
                        help="Validate every N epochs")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="anatomical-ldm",
                        help="Weights & Biases project name")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        mask_dir=args.mask_dir,
        val_mask_dir=args.val_mask_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create VAE
    vae = create_anatomical_vae(
        image_size=args.image_size,
        latent_channels=args.latent_channels,
        num_anatomical_regions=args.num_anatomical_regions,
        anatomical_loss_weight=args.anatomical_loss_weight,
    )
    
    # Create trainer
    trainer = VAETrainer(
        vae=vae,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_warmup_steps=args.beta_warmup_steps,
        anatomical_loss_weight=args.anatomical_loss_weight,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        validate_every=args.validate_every,
    )


if __name__ == "__main__":
    main()