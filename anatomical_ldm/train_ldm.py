"""
Training script for Anatomical Latent Diffusion Model.
Trains the UNet with anatomical conditioning in latent space.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb

from diffusers import DDPMScheduler, DDIMScheduler, DiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer

from .vae import AnatomicalVAE
from .anatomical_unet import AnatomicalUNet2DConditionModel, create_anatomical_unet

logger = logging.getLogger(__name__)


class LDMDataset(Dataset):
    """
    Dataset for LDM training with optional text conditioning.
    """
    
    def __init__(
        self,
        image_dir: str,
        text_file: Optional[str] = None,
        image_size: int = 512,
        extensions: Tuple[str] = ('.png', '.jpg', '.jpeg'),
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.glob(f"**/*{ext}")))
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Load text captions if provided
        self.texts = None
        if text_file:
            with open(text_file, 'r') as f:
                text_data = json.load(f)
            
            # Map image paths to text
            self.texts = {}
            for image_path in self.image_paths:
                # Use filename as key
                key = image_path.stem
                if key in text_data:
                    self.texts[str(image_path)] = text_data[key]
                else:
                    self.texts[str(image_path)] = ""  # Empty text for unconditional
            
            logger.info(f"Loaded texts for {len([t for t in self.texts.values() if t])} images")
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        result = {'image': image}
        
        # Add text if available
        if self.texts:
            text = self.texts.get(str(image_path), "")
            result['text'] = text
        
        return result


class AnatomicalLDMPipeline(DiffusionPipeline):
    """
    Custom pipeline for anatomical LDM inference.
    """
    
    def __init__(
        self,
        vae: AnatomicalVAE,
        unet: AnatomicalUNet2DConditionModel,
        scheduler: DDIMScheduler,
        text_encoder: Optional[CLIPTextModel] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        if self.text_encoder is None or self.tokenizer is None:
            return None
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
    ):
        """Generate images with anatomical conditioning."""
        
        # Text conditioning
        if prompt is not None and self.text_encoder is not None:
            text_embeddings = self.encode_text(prompt)
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                negative_text = negative_prompt or ""
                negative_embeddings = self.encode_text(negative_text)
                text_embeddings = torch.cat([negative_embeddings, text_embeddings])
            
            batch_size = text_embeddings.shape[0]
        else:
            text_embeddings = None
            batch_size = num_images_per_prompt
        
        # Prepare latents
        latent_height = height // self.vae.config.scaling_factor
        latent_width = width // self.vae.config.scaling_factor
        
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
        )
        
        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for t in tqdm(timesteps, desc="Generating"):
            # Expand latents for classifier-free guidance
            if guidance_scale > 1.0 and text_embeddings is not None:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )[0]
            
            # Classifier-free guidance
            if guidance_scale > 1.0 and text_embeddings is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to images
        images = self.vae.decode(latents, return_dict=False)[0]
        
        # Convert to PIL
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        
        return images


class LDMTrainer:
    """
    Trainer for Anatomical Latent Diffusion Model.
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
        output_dir: str = 'outputs/ldm',
        learning_rate: float = 1e-4,
        gradient_clip: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_wandb: bool = False,
        wandb_project: str = "anatomical-ldm",
        unconditional_prob: float = 0.1,  # Probability of unconditional training
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
        self.unconditional_prob = unconditional_prob
        
        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Freeze text encoder if used
        if self.text_encoder:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.unet.parameters(),
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
            wandb.init(project=wandb_project, name="ldm-training")
            wandb.config.update({
                "learning_rate": learning_rate,
                "unconditional_prob": unconditional_prob,
                "use_ema": use_ema,
            })
        
        self.global_step = 0
        self.epoch = 0
    
    def encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        """Encode text prompts to embeddings."""
        if self.text_encoder is None or self.tokenizer is None:
            return None
        
        # Tokenize
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        return text_embeddings
    
    def train_step(self, batch: Dict[str, any]) -> Dict[str, float]:
        """Single training step."""
        self.unet.train()
        
        images = batch['image'].to(self.device)
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
            
            # Randomly drop text for unconditional training (classifier-free guidance)
            if self.unconditional_prob > 0:
                for i in range(len(texts)):
                    if torch.rand(1).item() < self.unconditional_prob:
                        texts[i] = ""
            
            text_embeddings = self.encode_text(texts)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # EMA update
        if self.use_ema:
            self.ema_unet.step(self.unet.parameters())
        
        return {"loss": loss.item()}
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_dataloader is None:
            return {}
        
        self.unet.eval()
        total_loss = 0
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
                
                # Add noise
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
                total_loss += loss.item()
                num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def save_checkpoint(self, epoch: int, save_path: Optional[str] = None):
        """Save model checkpoint."""
        if save_path is None:
            save_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        
        # Get model to save (EMA if available)
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
        
        # Create inference scheduler
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            beta_start=self.noise_scheduler.config.beta_start,
            beta_end=self.noise_scheduler.config.beta_end,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            prediction_type=self.noise_scheduler.config.prediction_type,
        )
        
        # Create pipeline
        pipeline = AnatomicalLDMPipeline(
            vae=self.vae,
            unet=unet_to_save,
            scheduler=inference_scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )
        
        # Save
        pipeline.save_pretrained(save_path)
        logger.info(f"Saved pipeline to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema_unet.load_state_dict(checkpoint['ema_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate_samples(
        self,
        prompts: Optional[List[str]] = None,
        num_samples: int = 4,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate samples for visualization."""
        
        # Use EMA model if available
        unet_for_inference = self.ema_unet.averaged_model if self.use_ema else self.unet
        
        # Create inference scheduler
        inference_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            beta_start=self.noise_scheduler.config.beta_start,
            beta_end=self.noise_scheduler.config.beta_end,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
        )
        
        # Create pipeline
        pipeline = AnatomicalLDMPipeline(
            vae=self.vae,
            unet=unet_for_inference,
            scheduler=inference_scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )
        pipeline = pipeline.to(self.device)
        
        # Generate samples
        with torch.no_grad():
            if prompts:
                images = []
                for prompt in prompts[:num_samples]:
                    image = pipeline(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        height=512,
                        width=512,
                    )
                    images.append(image)
                samples = torch.cat(images, dim=0)
            else:
                samples = pipeline(
                    prompt=None,
                    num_images_per_prompt=num_samples,
                    num_inference_steps=num_inference_steps,
                    height=512,
                    width=512,
                )
        
        return samples
    
    def train(self, num_epochs: int, save_every: int = 10, validate_every: int = 5):
        """Main training loop."""
        logger.info(f"Starting LDM training for {num_epochs} epochs")
        
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
                    'loss': f"{step_losses['loss']:.4f}",
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
                
                # Generate samples
                try:
                    # Try with and without text
                    prompts = [
                        "a medical image showing normal anatomy",
                        "medical scan with clear anatomical structures", 
                        None,  # Unconditional
                        None,  # Unconditional
                    ]
                    
                    samples = self.generate_samples(prompts=prompts, num_samples=4)
                    sample_path = self.output_dir / f"samples_epoch_{epoch+1}.png"
                    save_image(
                        samples,
                        sample_path,
                        nrow=2,
                        normalize=True,
                        value_range=(0, 1)
                    )
                    
                    if self.use_wandb:
                        wandb.log({
                            "samples": wandb.Image(str(sample_path))
                        }, step=self.global_step)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate samples: {e}")
            
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
    parser = argparse.ArgumentParser(description="Train Anatomical LDM")
    
    # Data arguments
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Directory containing training images")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Directory containing validation images")
    parser.add_argument("--text_file", type=str, default=None,
                        help="JSON file with text captions (optional)")
    
    # Model arguments
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to pretrained VAE")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size")
    parser.add_argument("--latent_channels", type=int, default=4,
                        help="Number of latent channels")
    parser.add_argument("--cross_attention_dim", type=int, default=768,
                        help="Cross-attention dimension")
    parser.add_argument("--anatomical_conditioning_dim", type=int, default=512,
                        help="Anatomical conditioning dimension")
    parser.add_argument("--num_organs", type=int, default=12,
                        help="Number of anatomical organs")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--unconditional_prob", type=float, default=0.1,
                        help="Probability of unconditional training")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA for model weights")
    
    # Text conditioning
    parser.add_argument("--use_text_conditioning", action="store_true",
                        help="Enable text conditioning")
    parser.add_argument("--text_encoder_model", type=str, 
                        default="openai/clip-vit-base-patch32",
                        help="Text encoder model")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/ldm",
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
    
    # Load VAE
    logger.info(f"Loading VAE from {args.vae_path}")
    vae = AnatomicalVAE.from_pretrained(args.vae_path)
    
    # Create datasets
    train_dataset = LDMDataset(
        image_dir=args.train_dir,
        text_file=args.text_file if args.use_text_conditioning else None,
        image_size=args.image_size,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataloader = None
    if args.val_dir:
        val_dataset = LDMDataset(
            image_dir=args.val_dir,
            text_file=args.text_file if args.use_text_conditioning else None,
            image_size=args.image_size,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    # Create UNet
    unet = create_anatomical_unet(
        image_size=args.image_size,
        latent_channels=args.latent_channels,
        cross_attention_dim=args.cross_attention_dim,
        anatomical_conditioning_dim=args.anatomical_conditioning_dim,
        num_organs=args.num_organs,
    )
    
    # Text encoder
    text_encoder = None
    tokenizer = None
    if args.use_text_conditioning:
        from transformers import CLIPTextModel, CLIPTokenizer
        text_encoder = CLIPTextModel.from_pretrained(args.text_encoder_model)
        tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder_model)
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    
    # Create trainer
    trainer = LDMTrainer(
        unet=unet,
        vae=vae,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=args.device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        use_ema=args.use_ema,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        unconditional_prob=args.unconditional_prob,
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