#!/usr/bin/env python3
"""
Quick retraining script with fixed gate to test if it improves performance.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import diffusers
from diffusers.optimization import get_cosine_schedule_with_warmup

from fix_collapsed_gate import load_and_fix_model, replace_gate_entirely
from training import TrainingConfig
import datasets
from torchvision import transforms

def create_simple_dataset(img_dir, img_size=64, batch_size=32):
    """Create a simple dataset for testing."""
    from pathlib import Path
    
    # Find images
    img_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        img_paths.extend(list(Path(img_dir).glob(f"**/{ext}")))
    
    print(f"Found {len(img_paths)} images")
    
    # Create HF dataset
    dataset_dict = {"image": [str(p) for p in img_paths[:1000]]}  # Limit for quick test
    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("image", datasets.Image())
    
    # Transforms
    def transform(examples):
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
        
        images = [preprocess(image.convert('RGB')) for image in examples["image"]]
        return {"images": images}
    
    dataset.set_transform(transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def retrain_fixed_gate(
    checkpoint_dir, 
    img_dir, 
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda'
):
    """Retrain model with fixed gate."""
    
    print("Loading model with gate fix...")
    model = load_and_fix_model(checkpoint_dir, device)
    
    # Option: Replace gate entirely (more aggressive fix)
    # replace_gate_entirely(model)
    
    model = nn.DataParallel(model)
    model.train()
    
    # Create dataset
    print("Creating dataset...")
    dataloader = create_simple_dataset(img_dir, img_size=64, batch_size=batch_size)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_epochs * len(dataloader)
    )
    
    # Noise scheduler
    noise_scheduler = diffusers.DDIMScheduler(num_train_timesteps=1000)
    
    print(f"Starting retraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        gate_values = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            images = batch['images'].to(device)
            
            # Add noise
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Forward pass
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Track gate values
            with torch.no_grad():
                if hasattr(model.module, 'register_bank'):
                    register_dict = model.module.register_bank(images[:1], timesteps[:1])
                    pooled = register_dict["registers"].mean(dim=1)
                    gate_val = model.module.gate(pooled)
                    gate_values.append(gate_val.item())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gate': f'{gate_values[-1] if gate_values else 0:.3f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_gate = sum(gate_values) / len(gate_values) if gate_values else 0
        gate_std = torch.tensor(gate_values).std().item() if len(gate_values) > 1 else 0
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average gate: {avg_gate:.4f}")
        print(f"  Gate std: {gate_std:.4f}")
        
        # Check if gate is becoming more variable
        if gate_std > 0.01:
            print(f"  ✓ Gate is becoming variable! (std={gate_std:.4f})")
        else:
            print(f"  ⚠ Gate still not variable (std={gate_std:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = f"{checkpoint_dir}_fixed_epoch_{epoch+1}"
            os.makedirs(save_path, exist_ok=True)
            
            # Save the base UNet
            if hasattr(model.module, 'unet'):
                unet = model.module.unet
            else:
                unet = model.module
            
            pipeline = diffusers.DDIMPipeline(unet=unet, scheduler=noise_scheduler)
            pipeline.save_pretrained(save_path)
            
            # Save registers if available
            if hasattr(model.module, 'register_bank'):
                register_state = {
                    'register_bank': model.module.register_bank.state_dict(),
                    'register_proj': model.module.register_proj.state_dict(),
                    'gate': model.module.gate.state_dict(),
                }
                torch.save(register_state, os.path.join(save_path, 'anatomical_registers.pt'))
            
            print(f"  Saved checkpoint to {save_path}")
    
    print("\nRetraining complete!")
    print("Run quick_gate_check.py on the new checkpoint to see if gate behavior improved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    retrain_fixed_gate(
        args.checkpoint_dir,
        args.img_dir,
        args.num_epochs,
        args.batch_size,
        args.learning_rate
    )