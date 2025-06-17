"""
Evaluation and sampling utilities for Anatomical LDM.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore

from .train_ldm import AnatomicalLDMPipeline, LDMDataset

logger = logging.getLogger(__name__)


class AnatomicalMetrics:
    """
    Metrics for evaluating anatomical quality of generated chest X-rays.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Standard image quality metrics
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        self.inception_score = InceptionScore(normalize=True).to(device)
        
        # Optional: Load a pretrained anatomical segmentation model for anatomical metrics
        self.anatomical_segmentation_model = None
        self._load_anatomical_model()
    
    def _load_anatomical_model(self):
        """Load pretrained anatomical segmentation model if available."""
        try:
            # Try to load a pretrained model for anatomical evaluation
            # This could be a model trained to segment chest X-ray anatomy
            # For now, we'll skip this and focus on standard metrics
            pass
        except Exception as e:
            logger.warning(f"Could not load anatomical segmentation model: {e}")
    
    def compute_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Compute FID score."""
        # Convert grayscale to RGB for FID computation
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Ensure values are in [0, 1]
        real_images = (real_images + 1.0) / 2.0
        fake_images = (fake_images + 1.0) / 2.0
        
        # Convert to uint8
        real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
        fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
        
        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        
        return self.fid.compute().item()
    
    def compute_lpips(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Compute LPIPS score."""
        # Convert grayscale to RGB for LPIPS
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Ensure values are in [-1, 1] for LPIPS
        real_images = torch.clamp(real_images, -1, 1)
        fake_images = torch.clamp(fake_images, -1, 1)
        
        return self.lpips(real_images, fake_images).mean().item()
    
    def compute_inception_score(self, fake_images: torch.Tensor) -> Tuple[float, float]:
        """Compute Inception Score."""
        # Convert grayscale to RGB
        if fake_images.shape[1] == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Ensure values are in [0, 1]
        fake_images = (fake_images + 1.0) / 2.0
        fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
        
        self.inception_score.update(fake_images)
        is_mean, is_std = self.inception_score.compute()
        
        return is_mean.item(), is_std.item()
    
    def compute_anatomical_consistency(
        self, 
        images: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, float]:
        """
        Compute anatomical consistency metrics.
        This would use a pretrained anatomical segmentation model.
        """
        if self.anatomical_segmentation_model is None:
            return {"anatomical_consistency": 0.0}
        
        # TODO: Implement anatomical consistency evaluation
        # This would involve:
        # 1. Segmenting anatomical structures
        # 2. Checking for anatomical plausibility
        # 3. Measuring organ size ratios
        # 4. Evaluating spatial relationships
        
        return {"anatomical_consistency": 0.0}
    
    def compute_diversity(self, images: torch.Tensor) -> float:
        """Compute diversity score as average pairwise LPIPS."""
        batch_size = images.shape[0]
        if batch_size < 2:
            return 0.0
        
        # Convert to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Compute pairwise LPIPS
        diversity_scores = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                score = self.lpips(images[i:i+1], images[j:j+1])
                diversity_scores.append(score.item())
        
        return np.mean(diversity_scores) if diversity_scores else 0.0


class LDMEvaluator:
    """
    Comprehensive evaluator for Anatomical LDM.
    """
    
    def __init__(
        self,
        pipeline: AnatomicalLDMPipeline,
        device: str = 'cuda',
        output_dir: str = 'evaluation_results',
    ):
        self.pipeline = pipeline.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.metrics = AnatomicalMetrics(device)
    
    def generate_samples(
        self,
        num_samples: int = 100,
        prompts: Optional[List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        save_images: bool = True,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Generate samples for evaluation."""
        logger.info(f"Generating {num_samples} samples...")
        
        all_samples = []
        
        # Generate in batches
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            batch_end = min(i + batch_size, num_samples)
            current_batch_size = batch_end - i
            
            # Select prompts for this batch
            if prompts:
                batch_prompts = [
                    prompts[j % len(prompts)] for j in range(i, batch_end)
                ]
            else:
                batch_prompts = None
            
            # Generate
            with torch.no_grad():
                if batch_prompts:
                    batch_samples = []
                    for prompt in batch_prompts:
                        sample = self.pipeline(
                            prompt=prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=512,
                            width=512,
                        )
                        batch_samples.append(sample)
                    batch_samples = torch.cat(batch_samples, dim=0)
                else:
                    batch_samples = self.pipeline(
                        prompt=None,
                        num_images_per_prompt=current_batch_size,
                        num_inference_steps=num_inference_steps,
                        height=512,
                        width=512,
                    )
            
            all_samples.append(batch_samples.cpu())
            
            # Save individual batch if requested
            if save_images:
                batch_dir = self.output_dir / "generated_samples"
                batch_dir.mkdir(exist_ok=True)
                
                for j, sample in enumerate(batch_samples):
                    sample_idx = i + j
                    save_image(
                        sample,
                        batch_dir / f"sample_{sample_idx:05d}.png",
                        normalize=True,
                        value_range=(0, 1),
                    )
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        
        # Save grid
        if save_images:
            grid_path = self.output_dir / "sample_grid.png"
            save_image(
                all_samples[:64],  # First 64 samples
                grid_path,
                nrow=8,
                normalize=True,
                value_range=(0, 1),
            )
            logger.info(f"Saved sample grid to {grid_path}")
        
        return all_samples
    
    def evaluate_against_dataset(
        self,
        real_dataset: DataLoader,
        num_samples: int = 1000,
        prompts: Optional[List[str]] = None,
        num_inference_steps: int = 50,
    ) -> Dict[str, float]:
        """Evaluate generated samples against real dataset."""
        logger.info("Evaluating against real dataset...")
        
        # Generate samples
        fake_samples = self.generate_samples(
            num_samples=num_samples,
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            save_images=True,
        )
        
        # Load real samples
        real_samples = []
        for batch in tqdm(real_dataset, desc="Loading real samples"):
            real_samples.append(batch['image'])
            if len(real_samples) * batch['image'].shape[0] >= num_samples:
                break
        
        real_samples = torch.cat(real_samples, dim=0)[:num_samples]
        
        # Move to device for evaluation
        real_samples = real_samples.to(self.device)
        fake_samples = fake_samples.to(self.device)
        
        # Compute metrics
        results = {}
        
        # FID
        try:
            fid_score = self.metrics.compute_fid(real_samples, fake_samples)
            results['fid'] = fid_score
            logger.info(f"FID: {fid_score:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute FID: {e}")
        
        # LPIPS (on subset for efficiency)
        try:
            subset_size = min(100, len(real_samples), len(fake_samples))
            lpips_score = self.metrics.compute_lpips(
                real_samples[:subset_size],
                fake_samples[:subset_size]
            )
            results['lpips'] = lpips_score
            logger.info(f"LPIPS: {lpips_score:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute LPIPS: {e}")
        
        # Inception Score
        try:
            is_mean, is_std = self.metrics.compute_inception_score(fake_samples)
            results['inception_score_mean'] = is_mean
            results['inception_score_std'] = is_std
            logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute Inception Score: {e}")
        
        # Diversity
        try:
            diversity = self.metrics.compute_diversity(fake_samples[:50])  # Subset for efficiency
            results['diversity'] = diversity
            logger.info(f"Diversity: {diversity:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute diversity: {e}")
        
        # Anatomical consistency
        try:
            anatomical_metrics = self.metrics.compute_anatomical_consistency(fake_samples)
            results.update(anatomical_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute anatomical metrics: {e}")
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {results_path}")
        return results
    
    def compare_conditioning_strategies(
        self,
        prompts: List[str],
        anatomical_scales: List[float] = [0.0, 0.5, 1.0, 1.5],
        num_inference_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Compare different anatomical conditioning strategies."""
        logger.info("Comparing conditioning strategies...")
        
        results = {}
        
        for scale in anatomical_scales:
            logger.info(f"Testing anatomical conditioning scale: {scale}")
            
            # Set anatomical conditioning scale
            self.pipeline.unet.set_anatomical_conditioning_scale(scale)
            
            # Generate samples
            samples = []
            for prompt in prompts:
                sample = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=512,
                    width=512,
                )
                samples.append(sample)
            
            results[f"scale_{scale}"] = torch.cat(samples, dim=0)
        
        # Save comparison grid
        all_samples = []
        for scale in anatomical_scales:
            all_samples.append(results[f"scale_{scale}"])
        
        comparison = torch.cat(all_samples, dim=0)
        grid_path = self.output_dir / "conditioning_comparison.png"
        save_image(
            comparison,
            grid_path,
            nrow=len(prompts),
            normalize=True,
            value_range=(0, 1),
        )
        
        logger.info(f"Saved conditioning comparison to {grid_path}")
        return results
    
    def ablation_study(
        self,
        test_prompts: List[str],
        num_inference_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Perform ablation study on anatomical components."""
        logger.info("Performing ablation study...")
        
        results = {}
        
        # 1. Full model
        samples_full = self.generate_samples(
            num_samples=len(test_prompts),
            prompts=test_prompts,
            num_inference_steps=num_inference_steps,
            save_images=False,
        )
        results["full_model"] = samples_full
        
        # 2. Without anatomical conditioning (scale = 0)
        self.pipeline.unet.set_anatomical_conditioning_scale(0.0)
        samples_no_anatomical = self.generate_samples(
            num_samples=len(test_prompts),
            prompts=test_prompts,
            num_inference_steps=num_inference_steps,
            save_images=False,
        )
        results["no_anatomical"] = samples_no_anatomical
        
        # Reset to default
        self.pipeline.unet.set_anatomical_conditioning_scale(1.0)
        
        # Save comparison
        comparison = torch.cat([
            results["full_model"],
            results["no_anatomical"],
        ], dim=0)
        
        grid_path = self.output_dir / "ablation_study.png"
        save_image(
            comparison,
            grid_path,
            nrow=len(test_prompts),
            normalize=True,
            value_range=(0, 1),
        )
        
        logger.info(f"Saved ablation study to {grid_path}")
        return results


def create_evaluation_prompts() -> List[str]:
    """Create a set of evaluation prompts for chest X-rays."""
    return [
        "a normal chest X-ray with clear lung fields",
        "chest radiograph showing healthy lungs and heart",
        "frontal chest X-ray with normal cardiac silhouette",
        "chest X-ray with well-defined diaphragm and ribs",
        "normal posteroanterior chest radiograph",
        "chest X-ray showing symmetric lung fields",
        "clear chest radiograph with normal mediastinum",
        "chest X-ray with normal pulmonary vasculature",
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anatomical LDM")
    
    # Model arguments
    parser.add_argument("--pipeline_path", type=str, required=True,
                        help="Path to trained pipeline")
    parser.add_argument("--real_data_dir", type=str, required=True,
                        help="Directory containing real images for comparison")
    
    # Evaluation arguments
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    
    # Evaluation modes
    parser.add_argument("--run_full_evaluation", action="store_true",
                        help="Run full evaluation against real dataset")
    parser.add_argument("--run_conditioning_comparison", action="store_true",
                        help="Compare conditioning strategies")
    parser.add_argument("--run_ablation_study", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--generate_samples_only", action="store_true",
                        help="Only generate samples")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load pipeline
    logger.info(f"Loading pipeline from {args.pipeline_path}")
    pipeline = AnatomicalLDMPipeline.from_pretrained(args.pipeline_path)
    
    # Create evaluator
    evaluator = LDMEvaluator(
        pipeline=pipeline,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Create evaluation prompts
    eval_prompts = create_evaluation_prompts()
    
    if args.generate_samples_only:
        # Just generate samples
        evaluator.generate_samples(
            num_samples=args.num_samples,
            prompts=eval_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
        )
    
    elif args.run_full_evaluation:
        # Full evaluation against real dataset
        real_dataset = LDMDataset(
            image_dir=args.real_data_dir,
            image_size=512,
        )
        
        real_dataloader = DataLoader(
            real_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        
        results = evaluator.evaluate_against_dataset(
            real_dataset=real_dataloader,
            num_samples=args.num_samples,
            prompts=eval_prompts,
            num_inference_steps=args.num_inference_steps,
        )
        
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
    
    elif args.run_conditioning_comparison:
        # Compare conditioning strategies
        evaluator.compare_conditioning_strategies(
            prompts=eval_prompts[:4],  # Use subset for comparison
            num_inference_steps=args.num_inference_steps,
        )
    
    elif args.run_ablation_study:
        # Ablation study
        evaluator.ablation_study(
            test_prompts=eval_prompts[:4],
            num_inference_steps=args.num_inference_steps,
        )
    
    else:
        # Default: generate samples and run basic evaluation
        evaluator.generate_samples(
            num_samples=min(100, args.num_samples),
            prompts=eval_prompts,
            num_inference_steps=args.num_inference_steps,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()