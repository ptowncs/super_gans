"""
Metrics computation functions for FID and KID evaluation.
"""

import os
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

import super_gans.config as cfg
from pytorch_fid import fid_score


def calc_fid_score(real_images_dir, generated_images_dir):
    """
    Compute FID (Fréchet Inception Distance) between two directories of images.
    Uses pytorch_fid package.

    Args:
        real_images_dir: Path to directory with real images
        generated_images_dir: Path to directory with generated images

    Returns:
        float: FID score
    """
    # Paths to your image directories
    if not os.path.exists(real_images_dir):
        raise ValueError(f"Real image path not found: {real_images_dir}")
    if not os.path.exists(generated_images_dir):
        raise ValueError(f"Generated image path not found: {generated_images_dir}")

    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size=50,  # Adjust batch size based on available GPU memory
        device=cfg.device,
        dims=cfg.fid_dims,  # Inception v3 output dimension
    )
    return fid_value


def calc_kid_score(real_images_dir, generated_images_dir, subset_size=100):
    """
    Compute KID (Kernel Inception Distance) between two directories of images.
    Uses torchmetrics.KernelInceptionDistance.

    Args:
        real_images_dir: Path to directory with real images
        generated_images_dir: Path to directory with generated images
        subset_size: Size of subsets for KID computation (default: 100)

    Returns:
        tuple: (kid_mean, kid_std) - KID mean and standard deviation
    """
    # Define the transform: convert PIL image to tensor in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    real_dataset = ImageFolder(real_images_dir, transform=transform)
    generated_dataset = ImageFolder(generated_images_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=cfg.batch_size, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=cfg.batch_size, shuffle=False)

    kid_metric = KernelInceptionDistance(subset_size=subset_size).to(cfg.device)

    # Process real images
    for batch in real_loader:
        images = batch[0].to(cfg.device)
        kid_metric.update(images, real=True)

    # Process generated images
    for batch in generated_loader:
        images = batch[0].to(cfg.device)
        kid_metric.update(images, real=False)

    kid_mean, kid_std = kid_metric.compute()
    return kid_mean.item(), kid_std.item()