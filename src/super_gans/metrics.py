"""
Metrics computation functions for FID and KID evaluation.
"""

import os
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import super_gans.config as cfg
from pytorch_fid import fid_score
from PIL import Image


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
    Works with flat directories (no class subfolders required).

    Args:
        real_images_dir: Path to directory with real images
        generated_images_dir: Path to directory with generated images
        subset_size: Size of subsets for KID computation (default: 100)

    Returns:
        tuple: (kid_mean, kid_std) - KID mean and standard deviation
    """
    # Define the transform: convert PIL image to tensor in [0, 255] as uint8 for KID metric
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    class FlatFolderDataset(Dataset):
        # Custom dataset to avoid ImageFolder's requirement for class subfolders
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.paths = [
                os.path.join(root, f)
                for f in os.listdir(root)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Return a dummy label; KID metric only needs the image tensor.
            return image, 0

    real_dataset = FlatFolderDataset(real_images_dir, transform=transform)
    generated_dataset = FlatFolderDataset(generated_images_dir, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=cfg.batch_size, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=cfg.batch_size, shuffle=False)

    kid_metric = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(cfg.device)

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


def compute_fid_from_real_and_input(gen, loader, fid_metric):
    """
    Computes FID score by comparing real images from the loader
    with generated images from the generator.
    Expects loader to yield tuples of (real_images, generator_input).
    """
    gen.eval()
    fid_metric.reset()

    # Calculate how many batches we need to reach num_samples
    assert cfg.num_images_fid_sample % cfg.batch_size == 0, "FID sample count must be divisible by batch size"
    batch_size = cfg.batch_size
    n_batches = cfg.num_images_fid_sample // batch_size
    data_iter = iter(loader)

    # Ensure the metric is on the correct device
    fid_metric = fid_metric.to(cfg.device)

    with torch.inference_mode():
        for _ in range(n_batches):
            # --- 1. Process Real Images ---
            try:
                real_batch, gen_input_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_batch, gen_input_batch = next(data_iter)

            real_batch = real_batch[:batch_size].to(cfg.device)
            gen_input_batch = gen_input_batch[:batch_size].to(cfg.device)

            # Map [-1, 1] -> [0, 1] and expand grayscale to 3 channels
            real_rgb = (real_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0

            fid_metric.update(real_rgb, real=True)

            # --- 2. Process Fake Images ---
            fake_batch = gen(gen_input_batch)
            fake_rgb = (fake_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0

            fid_metric.update(fake_rgb, real=False)

        # --- 3. Compute and Log ---
        fid_score = fid_metric.compute().item()

        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        gen.train()
        return fid_score


def compute_fid_from_images(gen, loader, fid_metric):
    """
    Computes FID score using a standard data loader (yielding only real images).
    Wraps the loader to provide noise as generator input for standard GANs.
    """
    def wrapped_loader():
        for real_batch, _ in loader:  # Assuming loader yields (image, label) or similar
            noise = torch.randn(real_batch.size(0), cfg.z_dim, device=cfg.device)
            yield real_batch, noise

    return compute_fid_from_real_and_input(gen, wrapped_loader(), fid_metric)