"""
Training script for Supervised Anatomical LDM with mask supervision.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb

from diffusers import DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from .vae import AnatomicalVAE
from .anatomical_unet import AnatomicalUNet2DConditionModel
from .supervised_registers import SupervisedAnatomicalRegisterBank
from .train_ldm import AnatomicalLDMPipeline

logger = logging.getLogger(__name__)


class SupervisedLDMDataset(Dataset):
    """
    Dataset for supervised LDM training with anatomical masks.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        text_file: Optional[str] = None,
        image_size: int = 512,
        extensions: Tuple[str] = ('.png', '.jpg', '.jpeg'),
        mask_extensions: Tuple[str] = ('.png', '.npy'),
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.glob(f"**/*{ext}")))
        
        # Find corresponding masks
        self.mask_paths = []
        missing_masks = 0
        
        for image_path in self.image_paths:
            # Look for corresponding mask
            relative_path = image_path.relative_to(self.image_dir)
            
            mask_found = False
            for mask_ext in mask_extensions:
                mask_path = self.mask_dir / relative_path.with_suffix(mask_ext)
                if mask_path.exists():
                    self.mask_paths.append(mask_path)
                    mask_found = True
                    break
            
            if not mask_found:
                self.mask_paths.append(None)
                missing_masks += 1
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        logger.info(f"Found {len(self.image_paths) - missing_masks} corresponding masks")
        logger.info(f"Missing {missing_masks} masks")
        
        # Load text captions if provided
        self.texts = None
        if text_file:
            with open(text_file, 'r') as f:
                text_data = json.load(f)
            
            self.texts = {}
            for image_path in self.image_paths:
                key = image_path.stem
                if key in text_data:
                    self.texts[str(image_path)] = text_data[key]
                else:
                    self.texts[str(image_path)] = ""
        
        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        result = {'image': image}
        
        # Load mask if available
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            try:
                if mask_path.suffix == '.npy':
                    # NumPy array
                    mask = np.load(mask_path)
                    mask = torch.from_numpy(mask).long()
                else:
                    # Image file
                    mask = Image.open(mask_path)
                    mask = transforms.ToTensor()(mask)
                    mask = (mask * 255).long().squeeze(0)  # Convert to class indices
                
                # Resize mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                mask = F.interpolate(
                    mask.float().unsqueeze(0), 
                    size=(self.image_size, self.image_size), 
                    mode='nearest'
                ).squeeze().long()
                
                result['mask'] = mask
                
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e}")
        
        # Add text if available
        if self.texts:
            text = self.texts.get(str(image_path), "")
            result['text'] = text
        
        return result


class SupervisedLDMTrainer:
    """
    Trainer for Supervised Anatomical LDM with mask supervision.
    """
    
    def __init__(
        self,
        unet: AnatomicalUNet2DConditionModel,
        vae: AnatomicalVAE,
        noise_scheduler: DDPMScheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        device: str = 'cuda',
        output_dir: str = 'outputs/supervised_ldm',
        learning_rate: float = 1e-4,
        # Supervision parameters
        anatomical_supervision_weight: float = 1.0,
        use_anatomical_supervision: bool = True,
        supervision_probability: float = 1.0,  # Probability of using supervision when available
        # Other parameters
        gradient_clip: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_wandb: bool = False,
        wandb_project: str = "anatomical-ldm",
        unconditional_prob: float = 0.1,
    ):
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.text_encoder = text_encoder.to(device) if text_encoder else None
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supervision parameters
        self.anatomical_supervision_weight = anatomical_supervision_weight
        self.use_anatomical_supervision = use_anatomical_supervision
        self.supervision_probability = supervision_probability
        self.unconditional_prob = unconditional_prob
        
        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Freeze text encoder
        if self.text_encoder:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Optimizer - include register bank parameters
        all_params = list(self.unet.parameters())
        if hasattr(self.unet, 'anatomical_registers'):
            all_params.extend(self.unet.anatomical_registers.parameters())
        
        self.optimizer = AdamW(
            all_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # EMA
        self.use_ema = use_ema
        if use_ema:
            from diffusers.training_utils import EMAModel
            self.ema_unet = EMAModel(self.unet.parameters(), decay=ema_decay)
        
        self.gradient_clip = gradient_clip
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, name="supervised-ldm-training")
            wandb.config.update({
                "learning_rate": learning_rate,
                "anatomical_supervision_weight": anatomical_supervision_weight,
                "supervision_probability": supervision_probability,
                "use_anatomical_supervision": use_anatomical_supervision,
            })
        
        self.global_step = 0
        self.epoch = 0
    
    def encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Encode text prompts to embeddings."""
        if self.text_encoder is None or self.tokenizer is None:
            return None
        
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def train_step(self, batch: Dict[str, any]) -> Dict[str, float]:
        """Single training step with anatomical supervision."""
        self.unet.train()
        
        images = batch['image'].to(self.device)
        masks = batch.get('mask', None)
        batch_size = images.shape[0]
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images, return_dict=False)[0].sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Text conditioning
        text_embeddings = None
        if 'text' in batch and self.text_encoder is not None:
            texts = batch['text']
            
            # Randomly drop text for unconditional training
            if self.unconditional_prob > 0:
                for i in range(len(texts)):
                    if torch.rand(1).item() < self.unconditional_prob:
                        texts[i] = ""
            
            text_embeddings = self.encode_text(texts)
        
        # Key innovation: Anatomical supervision
        use_supervision = (
            self.use_anatomical_supervision and 
            masks is not None and 
            torch.rand(1).item() < self.supervision_probability
        )
        
        anatomical_supervision_loss = 0.0
        if use_supervision:
            masks = masks.to(self.device)
            
            # Prepare latent features for anatomical prediction
            # Use clean latents for anatomical learning (not noisy ones)
            if hasattr(self.unet, 'anatomical_registers') and isinstance(
                self.unet.anatomical_registers, SupervisedAnatomicalRegisterBank
            ):
                # Get anatomical predictions from register bank
                register_dict = self.unet.anatomical_registers(
                    batch_size=batch_size,
                    timestep=timesteps,
                    device=self.device,
                    latent_height=latents.shape[-2],
                    latent_width=latents.shape[-1],
                    latent_features=latents,  # Use clean latents for supervision
                )
                
                if "anatomical_predictions" in register_dict:
                    # Compute anatomical supervision loss
                    anatomical_loss_dict = self.unet.anatomical_registers.compute_anatomical_supervision_loss(
                        register_dict["anatomical_predictions"],
                        masks
                    )
                    anatomical_supervision_loss = anatomical_loss_dict["total_anatomical_loss"]
        
        # Standard diffusion prediction
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]
        
        # Compute losses
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        total_loss = diffusion_loss
        if use_supervision:
            total_loss = total_loss + self.anatomical_supervision_weight * anatomical_supervision_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # EMA update
        if self.use_ema:
            self.ema_unet.step(self.unet.parameters())
        
        # Return losses
        result = {
            "total_loss": total_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
        }
        
        if use_supervision:
            result["anatomical_supervision_loss"] = anatomical_supervision_loss.item()
            result["used_supervision"] = 1.0
        else:
            result["anatomical_supervision_loss"] = 0.0
            result["used_supervision"] = 0.0
        
        return result
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_dataloader is None:
            return {}
        
        self.unet.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                
                # Encode to latent space
                latents = self.vae.encode(images, return_dict=False)[0].sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=self.device, dtype=torch.long
                )
                
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text conditioning
                text_embeddings = None
                if 'text' in batch and self.text_encoder is not None:
                    text_embeddings = self.encode_text(batch['text'])
                
                # Predict noise
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Accumulate
                for k, v in {"val_diffusion_loss": loss.item()}.items():
                    if k not in total_losses:
                        total_losses[k] = 0
                    total_losses[k] += v
                
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_checkpoint(self, epoch: int, save_path: Optional[str] = None):
        """Save model checkpoint."""
        if save_path is None:
            save_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        
        unet_to_save = self.ema_unet.averaged_model if self.use_ema else self.unet
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'unet_state_dict': unet_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'unet_config': self.unet.config,
        }
        
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_unet.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        # Save pipeline
        pipeline_path = self.output_dir / f"pipeline_epoch_{epoch}"
        self.save_pipeline(pipeline_path, unet_to_save)
    
    def save_pipeline(self, save_path: Path, unet_to_save=None):
        """Save complete pipeline."""
        if unet_to_save is None:
            unet_to_save = self.ema_unet.averaged_model if self.use_ema else self.unet
        
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            beta_start=self.noise_scheduler.config.beta_start,
            beta_end=self.noise_scheduler.config.beta_end,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            prediction_type=self.noise_scheduler.config.prediction_type,
        )
        
        pipeline = AnatomicalLDMPipeline(
            vae=self.vae,
            unet=unet_to_save,
            scheduler=inference_scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )
        
        pipeline.save_pretrained(save_path)
        logger.info(f"Saved pipeline to {save_path}")
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 5):
        """Main training loop."""
        logger.info(f"Starting supervised LDM training for {num_epochs} epochs")
        
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
                    'total': f"{step_losses['total_loss']:.4f}",
                    'diff': f"{step_losses['diffusion_loss']:.4f}",
                    'anat': f"{step_losses['anatomical_supervision_loss']:.4f}",
                    'sup': f"{step_losses['used_supervision']:.1f}",
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


def main():
    parser = argparse.ArgumentParser(description="Train Supervised Anatomical LDM")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--val_mask_dir", type=str, default=None)
    parser.add_argument("--text_file", type=str, default=None)
    
    # Model arguments
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--cross_attention_dim", type=int, default=768)
    parser.add_argument("--anatomical_conditioning_dim", type=int, default=512)
    parser.add_argument("--num_organs", type=int, default=12)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # Supervision arguments
    parser.add_argument("--anatomical_supervision_weight", type=float, default=1.0)
    parser.add_argument("--supervision_probability", type=float, default=1.0)
    parser.add_argument("--use_anatomical_supervision", action="store_true")
    
    # Other arguments
    parser.add_argument("--use_text_conditioning", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/supervised_ldm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load VAE
    vae = AnatomicalVAE.from_pretrained(args.vae_path)
    
    # Create dataset
    train_dataset = SupervisedLDMDataset(
        image_dir=args.train_dir,
        mask_dir=args.mask_dir,
        text_file=args.text_file if args.use_text_conditioning else None,
        image_size=args.image_size,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Create UNet with supervised registers
    unet = AnatomicalUNet2DConditionModel(
        sample_size=args.image_size // 8,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        cross_attention_dim=args.cross_attention_dim,
        anatomical_conditioning_dim=args.anatomical_conditioning_dim,
        anatomical_num_organs=args.num_organs,
    )
    
    # Replace register bank with supervised version
    unet.anatomical_registers = SupervisedAnatomicalRegisterBank(
        d_model=args.anatomical_conditioning_dim,
        num_organs=args.num_organs,
        spatial_resolution=8,
    )
    
    # Text encoder
    text_encoder = None
    tokenizer = None
    if args.use_text_conditioning:
        from transformers import CLIPTextModel, CLIPTokenizer
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    # Create trainer
    trainer = SupervisedLDMTrainer(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        anatomical_supervision_weight=args.anatomical_supervision_weight,
        supervision_probability=args.supervision_probability,
        use_anatomical_supervision=args.use_anatomical_supervision,
        use_ema=args.use_ema,
        use_wandb=args.use_wandb,
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()