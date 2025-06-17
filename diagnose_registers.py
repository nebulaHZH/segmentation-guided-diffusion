#!/usr/bin/env python3
"""
Diagnostic tool to analyze learned anatomical registers and understand their usage.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import os
import diffusers
from anatomical_registers import AnatomicalRegisterBank, RegisterModulatedUNet
from training import TrainingConfig

def load_model_with_registers(checkpoint_dir, device='cuda'):
    """Load a model with anatomical registers."""
    # Load config
    config_path = Path(checkpoint_dir) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        # Default config for your training
        config_dict = {
            'register_dim': 512,
            'num_organ_registers': 12,
            'num_spatial_registers': 6,
            'num_scale_registers': 4
        }
    
    # Load base UNet
    unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(checkpoint_dir, 'unet'),
        use_safetensors=True
    )
    
    # Create register bank
    register_bank = AnatomicalRegisterBank(
        dim=config_dict.get('register_dim', 512),
        num_organ_registers=config_dict.get('num_organ_registers', 12),
        num_spatial_registers=config_dict.get('num_spatial_registers', 6),
        num_scale_registers=config_dict.get('num_scale_registers', 4)
    )
    
    # Wrap in RegisterModulatedUNet
    model = RegisterModulatedUNet(unet, register_bank)
    
    # Load register weights
    register_path = Path(checkpoint_dir) / 'anatomical_registers.pt'
    if register_path.exists():
        print(f"Loading anatomical registers from {register_path}")
        register_state = torch.load(register_path, map_location=device)
        model.register_bank.load_state_dict(register_state['register_bank'])
        model.register_proj.load_state_dict(register_state['register_proj'])
        model.gate.load_state_dict(register_state['gate'])
    else:
        print("WARNING: No anatomical_registers.pt found!")
    
    model.to(device)
    model.eval()
    return model

def analyze_gate_values(model, num_samples=1000, batch_size=32, img_size=64, device='cuda'):
    """Analyze what gate values the model has learned."""
    print("\n=== Analyzing Gate Values ===")
    
    all_gate_values = []
    all_timesteps = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Collecting gate values"):
            # Create random batch
            x = torch.randn(batch_size, 1, img_size, img_size).to(device)
            # Sample timesteps across full range
            timesteps = torch.linspace(0, 999, batch_size).long().to(device)
            
            # Get registers and compute gate
            register_dict = model.register_bank(x, timesteps)
            pooled_registers = register_dict["registers"].mean(dim=1)
            gate_value = model.gate(pooled_registers)
            
            all_gate_values.extend(gate_value.cpu().numpy().flatten())
            all_timesteps.extend(timesteps.cpu().numpy())
    
    all_gate_values = np.array(all_gate_values)
    all_timesteps = np.array(all_timesteps)
    
    # Statistics
    print(f"Gate value statistics:")
    print(f"  Mean: {np.mean(all_gate_values):.4f}")
    print(f"  Std: {np.std(all_gate_values):.4f}")
    print(f"  Min: {np.min(all_gate_values):.4f}")
    print(f"  Max: {np.max(all_gate_values):.4f}")
    print(f"  Median: {np.median(all_gate_values):.4f}")
    
    # Plot gate values vs timestep
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(all_timesteps, all_gate_values, alpha=0.5, s=10)
    plt.xlabel('Timestep')
    plt.ylabel('Gate Value')
    plt.title('Gate Values vs Timestep')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    window = 50
    timestep_bins = np.arange(0, 1000, window)
    mean_gates = []
    for t in timestep_bins:
        mask = (all_timesteps >= t) & (all_timesteps < t + window)
        if mask.sum() > 0:
            mean_gates.append(np.mean(all_gate_values[mask]))
        else:
            mean_gates.append(0)
    plt.plot(timestep_bins + window/2, mean_gates, 'r-', linewidth=2, label='Moving Average')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(all_gate_values, bins=50, alpha=0.7, density=True)
    plt.xlabel('Gate Value')
    plt.ylabel('Density')
    plt.title('Distribution of Gate Values')
    plt.axvline(np.mean(all_gate_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_gate_values):.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gate_analysis.png', dpi=150)
    plt.close()
    
    return all_gate_values, all_timesteps

def analyze_register_usage(model, device='cuda'):
    """Analyze which registers are being used and how."""
    print("\n=== Analyzing Register Usage ===")
    
    # Get register weights
    register_bank = model.register_bank
    mixing_weights = torch.softmax(register_bank.register_mixing, dim=0).detach().cpu().numpy()
    
    print(f"Register mixing weights:")
    print(f"  Organ registers: {mixing_weights[0]:.3f}")
    print(f"  Spatial registers: {mixing_weights[1]:.3f}")
    print(f"  Scale registers: {mixing_weights[2]:.3f}")
    
    # Analyze register magnitudes
    organ_mag = torch.norm(register_bank.organ_registers, dim=1).mean().item()
    spatial_mag = torch.norm(register_bank.spatial_registers, dim=1).mean().item()
    scale_mag = torch.norm(register_bank.scale_registers, dim=1).mean().item()
    
    print(f"\nRegister magnitudes (L2 norm):")
    print(f"  Organ registers: {organ_mag:.3f}")
    print(f"  Spatial registers: {spatial_mag:.3f}")
    print(f"  Scale registers: {scale_mag:.3f}")
    
    # Analyze register diversity (how different are registers within each type)
    def compute_diversity(registers):
        # Compute pairwise cosine similarities
        norms = torch.norm(registers, dim=1, keepdim=True)
        normalized = registers / (norms + 1e-8)
        similarities = torch.matmul(normalized, normalized.T)
        # Get off-diagonal elements
        n = similarities.shape[0]
        mask = ~torch.eye(n, dtype=bool)
        off_diag = similarities[mask]
        return off_diag.mean().item(), off_diag.std().item()
    
    organ_sim_mean, organ_sim_std = compute_diversity(register_bank.organ_registers)
    spatial_sim_mean, spatial_sim_std = compute_diversity(register_bank.spatial_registers)
    scale_sim_mean, scale_sim_std = compute_diversity(register_bank.scale_registers)
    
    print(f"\nRegister diversity (cosine similarity):")
    print(f"  Organ registers: {organ_sim_mean:.3f} ± {organ_sim_std:.3f}")
    print(f"  Spatial registers: {spatial_sim_mean:.3f} ± {spatial_sim_std:.3f}")
    print(f"  Scale registers: {scale_sim_mean:.3f} ± {scale_sim_std:.3f}")
    print("  (Lower similarity = more diverse)")
    
    # Visualize register similarities
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Organ registers
    organ_sims = torch.matmul(
        register_bank.organ_registers / torch.norm(register_bank.organ_registers, dim=1, keepdim=True),
        (register_bank.organ_registers / torch.norm(register_bank.organ_registers, dim=1, keepdim=True)).T
    ).cpu().numpy()
    
    sns.heatmap(organ_sims, ax=axes[0], cmap='coolwarm', center=0, 
                xticklabels=register_bank.organ_names, 
                yticklabels=register_bank.organ_names)
    axes[0].set_title('Organ Register Similarities')
    
    # Spatial registers
    spatial_sims = torch.matmul(
        register_bank.spatial_registers / torch.norm(register_bank.spatial_registers, dim=1, keepdim=True),
        (register_bank.spatial_registers / torch.norm(register_bank.spatial_registers, dim=1, keepdim=True)).T
    ).cpu().numpy()
    
    sns.heatmap(spatial_sims, ax=axes[1], cmap='coolwarm', center=0,
                xticklabels=['ant', 'post', 'left', 'right', 'sup', 'inf'],
                yticklabels=['ant', 'post', 'left', 'right', 'sup', 'inf'])
    axes[1].set_title('Spatial Register Similarities')
    
    # Scale registers
    scale_sims = torch.matmul(
        register_bank.scale_registers / torch.norm(register_bank.scale_registers, dim=1, keepdim=True),
        (register_bank.scale_registers / torch.norm(register_bank.scale_registers, dim=1, keepdim=True)).T
    ).cpu().numpy()
    
    sns.heatmap(scale_sims, ax=axes[2], cmap='coolwarm', center=0,
                xticklabels=['organ', 'tissue', 'cellular', 'fine'],
                yticklabels=['organ', 'tissue', 'cellular', 'fine'])
    axes[2].set_title('Scale Register Similarities')
    
    plt.tight_layout()
    plt.savefig('register_similarities.png', dpi=150)
    plt.close()

def analyze_stage_behavior(model, device='cuda'):
    """Analyze how registers behave at different diffusion stages."""
    print("\n=== Analyzing Stage-Specific Behavior ===")
    
    batch_size = 32
    x = torch.randn(batch_size, 1, 64, 64).to(device)
    
    # Test at different timesteps
    timestep_stages = {
        'early': torch.full((batch_size,), 900, device=device),
        'mid': torch.full((batch_size,), 500, device=device),
        'late': torch.full((batch_size,), 100, device=device)
    }
    
    stage_registers = {}
    for stage_name, timesteps in timestep_stages.items():
        registers = model.register_bank.get_stage_registers(timesteps, batch_size)
        stage_registers[stage_name] = registers.mean(dim=0).cpu().numpy()  # Average across batch
    
    # Visualize register activations at different stages
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (stage_name, registers) in enumerate(stage_registers.items()):
        # Split registers by type
        organ_end = 12
        spatial_end = organ_end + 6
        
        organ_regs = np.mean(np.abs(registers[:organ_end]), axis=1)
        spatial_regs = np.mean(np.abs(registers[organ_end:spatial_end]), axis=1)
        scale_regs = np.mean(np.abs(registers[spatial_end:]), axis=1)
        
        x_pos = np.arange(3)
        values = [organ_regs.mean(), spatial_regs.mean(), scale_regs.mean()]
        
        axes[idx].bar(x_pos, values, color=['blue', 'green', 'red'])
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(['Organ', 'Spatial', 'Scale'])
        axes[idx].set_ylabel('Average Activation')
        axes[idx].set_title(f'{stage_name.capitalize()} Stage (t={timesteps[0].item()})')
        axes[idx].set_ylim(0, max([v.max() for v in stage_registers.values()]) * 1.2)
    
    plt.suptitle('Register Type Activations by Diffusion Stage')
    plt.tight_layout()
    plt.savefig('stage_analysis.png', dpi=150)
    plt.close()

def analyze_gradient_flow(model, dataloader, device='cuda', num_batches=10):
    """Analyze if gradients are flowing to registers during training."""
    print("\n=== Analyzing Gradient Flow ===")
    
    model.train()
    criterion = nn.MSELoss()
    
    # Track gradients
    register_grads = []
    unet_grads = []
    gate_grads = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        images = batch['images'].to(device)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
        
        # Forward pass
        noisy_images = images + noise * 0.1  # Simplified for analysis
        output = model(noisy_images, timesteps, return_dict=False)[0]
        loss = criterion(output, noise)
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        if model.register_bank.organ_registers.grad is not None:
            register_grads.append(model.register_bank.organ_registers.grad.norm().item())
        
        if model.gate[0].weight.grad is not None:
            gate_grads.append(model.gate[0].weight.grad.norm().item())
            
        # Sample UNet gradient
        for name, param in model.unet.named_parameters():
            if param.grad is not None and 'conv' in name:
                unet_grads.append(param.grad.norm().item())
                break
        
        # Clear gradients
        model.zero_grad()
    
    print(f"\nGradient magnitudes (L2 norm):")
    if register_grads:
        print(f"  Register gradients: {np.mean(register_grads):.6f} ± {np.std(register_grads):.6f}")
    if gate_grads:
        print(f"  Gate gradients: {np.mean(gate_grads):.6f} ± {np.std(gate_grads):.6f}")
    if unet_grads:
        print(f"  UNet gradients: {np.mean(unet_grads):.6f} ± {np.std(unet_grads):.6f}")
    
    if not register_grads:
        print("  WARNING: No register gradients detected!")
    
    model.eval()

def compare_with_baseline(anatomical_dir, baseline_dir, num_samples=100, device='cuda'):
    """Compare outputs between anatomical and baseline models."""
    print("\n=== Comparing with Baseline ===")
    
    # Load baseline model
    baseline_unet = diffusers.UNet2DModel.from_pretrained(
        os.path.join(baseline_dir, 'unet'),
        use_safetensors=True
    ).to(device).eval()
    
    # Load anatomical model
    anatomical_model = load_model_with_registers(anatomical_dir, device)
    
    # Compare on same inputs
    differences = []
    gate_values = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Comparing models"):
            x = torch.randn(1, 1, 64, 64).to(device)
            t = torch.randint(0, 1000, (1,), device=device)
            
            # Baseline output
            baseline_out = baseline_unet(x, t, return_dict=True).sample
            
            # Anatomical output
            anatomical_out = anatomical_model(x, t, return_dict=True).sample
            
            # Get gate value for this sample
            register_dict = anatomical_model.register_bank(x, t)
            pooled = register_dict["registers"].mean(dim=1)
            gate = anatomical_model.gate(pooled)
            gate_values.append(gate.item())
            
            # Compute difference
            diff = torch.mean(torch.abs(anatomical_out - baseline_out)).item()
            differences.append(diff)
    
    differences = np.array(differences)
    gate_values = np.array(gate_values)
    
    print(f"\nOutput differences (L1):")
    print(f"  Mean: {np.mean(differences):.6f}")
    print(f"  Std: {np.std(differences):.6f}")
    print(f"  Max: {np.max(differences):.6f}")
    
    # Plot correlation between gate values and differences
    plt.figure(figsize=(8, 6))
    plt.scatter(gate_values, differences, alpha=0.5)
    plt.xlabel('Gate Value')
    plt.ylabel('L1 Difference from Baseline')
    plt.title('Gate Value vs Output Difference')
    
    # Add correlation coefficient
    corr = np.corrcoef(gate_values, differences)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('baseline_comparison.png', dpi=150)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose anatomical register usage')
    parser.add_argument('--anatomical_dir', type=str, required=True, 
                        help='Path to anatomical model checkpoint directory')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Path to baseline model checkpoint directory for comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_with_registers(args.anatomical_dir, args.device)
    
    # Run analyses
    gate_values, timesteps = analyze_gate_values(model, args.num_samples, device=args.device)
    analyze_register_usage(model, device=args.device)
    analyze_stage_behavior(model, device=args.device)
    
    # If baseline provided, compare
    if args.baseline_dir:
        compare_with_baseline(args.anatomical_dir, args.baseline_dir, 
                            num_samples=min(args.num_samples, 200), device=args.device)
    
    print("\n=== Analysis Complete ===")
    print("Generated plots:")
    print("  - gate_analysis.png: Gate value statistics and distribution")
    print("  - register_similarities.png: Register similarity matrices")
    print("  - stage_analysis.png: Stage-specific register activations")
    if args.baseline_dir:
        print("  - baseline_comparison.png: Comparison with baseline model")

if __name__ == "__main__":
    main()