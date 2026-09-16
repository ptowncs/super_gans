import gc  # Good practice for cleaning memory during training
import os
from pathlib import Path

import psutil  # Used for RAM tracking print statements

# PyTorch Core & Dataset Handling
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from torch.utils.data import ConcatDataset, Subset

# Evaluation Metrics (Inline Validation & Final Benchmarks)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

import super_gans.config as cfg
from super_gans.data_handling import (
    PneumoniaKaggleDataset,
    PneumoniaRsnaDataset,
    prepare_kaggle_data,
    prepare_rsna_data,
)


def create_dirs():
    dirs = [cfg.DATA_DIR, cfg.RESULTS_DIR, cfg.MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_transforms_pipeline():
    """Get the standard transforms pipeline for pneumonia dataset."""
    return transforms.Compose([
        transforms.Grayscale(
            num_output_channels=cfg.num_channels
        ),  # Force 1 channel
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),  # image to tensor
        transforms.Normalize((0.5,), (0.5,)),  # normalize images , [0,1] to [-1,1]
    ])


def load_data(split="train"):
    """
    Load pneumonia dataset for the specified split.
    Uses configurable data source (Kaggle or RSNA) via cfg.DATA_SOURCE.
    Loads ONLY pneumonia images (ignores normal).
    """
    # Ensure data is prepared (downloaded, split, and copied to split directories)
    data_source = cfg.DATA_SOURCE.lower()
    split_dir = "./data/split"
    transforms_pipeline = get_transforms_pipeline()

    if data_source == 'kaggle':
        prepare_kaggle_data("./data", split_dir)
        # Load from split directories (non-recursive - flattened structure)
        if split == "train":
            image_paths = [p for p in Path(split_dir).glob("train/*") if p.is_file()]
        elif split == "val" or split == "validation":
            image_paths = [p for p in Path(split_dir).glob("val/*") if p.is_file()]
        elif split == "test":
            # For simplicity, use validation set as test
            image_paths = [p for p in Path(split_dir).glob("val/*") if p.is_file()]
            print("Using validation set as test set")
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

        # Filter for valid image extensions
        valid_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        image_paths = [p for p in image_paths if p.suffix.lower() in valid_extensions]

        dataset = PneumoniaKaggleDataset(
            image_paths=image_paths,
            transform=transforms_pipeline,
        )
    elif data_source == 'rsna':
        prepare_rsna_data("./data", split_dir)
        # Load from split directories (non-recursive - flattened structure)
        if split == "train":
            image_paths = [p for p in Path(split_dir).glob("train/*") if p.is_file()]
        elif split == "val" or split == "validation":
            image_paths = [p for p in Path(split_dir).glob("val/*") if p.is_file()]
        elif split == "test":
            # For simplicity, use validation set as test
            image_paths = [p for p in Path(split_dir).glob("val/*") if p.is_file()]
            print("Using validation set as test set")
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

        # Filter for DICOM files
        image_paths = [p for p in image_paths if p.suffix.lower() == ".dcm"]

        dataset = PneumoniaRsnaDataset(
            image_paths=image_paths,
            transform=transforms_pipeline,
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}. Use 'kaggle' or 'rsna'")

    print(f"Loaded {len(dataset)} images for {split} split from {data_source} dataset")
    return dataset

def build_fid_evaluation_dataset(load_fn=load_data, target_samples=5000):
    val_ds = load_fn(split="val")
    test_ds = load_fn(split="test")
    
    eval_ds = ConcatDataset([val_ds, test_ds])
    current_count = len(eval_ds)
    print(f"Val + Test images: {current_count}")
    
    if current_count < target_samples:
        needed = target_samples - current_count
        print(f"Pulling the exact same {needed} sequential images from Train split...")
        
        train_ds = load_fn(split="train")
        
        # Always pick indices 0 to needed (guarantees the same images every time)
        deterministic_indices = list(range(needed))
        train_supplement = Subset(train_ds, deterministic_indices)
        
        eval_ds = ConcatDataset([eval_ds, train_supplement])
        
    print(f"Final reproducible FID dataset complete with {len(eval_ds)} images.")
    return eval_ds

def save_images_fid(dataset, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    for i in range(min(cfg.num_images_fid_score, len(dataset))):
        image, _ = dataset[i]  # image is a tensor in shape (1, 64, 64)

        # Convert grayscale -> RGB by repeating channels
        image_rgb = image.repeat(3, 1, 1)
        filename = os.path.join(to_dir, f"pneumonia_{i:04d}.png")
        # Generator uses Tanh (outputting [-1, 1]), ensure normalize=True and value_range=(-1, 1) 
        # so the PNGs are stored as standard [0, 255] pixel values correctly.
        save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))

    print(
        f"Saved {min(cfg.num_images_fid_score, len(dataset))} real Pneumonia images to {to_dir}/"
    )

def generate_images_fid(generator, generated_images_dir, batch_size=128):
    os.makedirs(generated_images_dir, exist_ok=True)
    generator.eval() # Ensure evaluation mode
    
    images_saved = 0
    # Process in smaller chunks to prevent CUDA OOM
    while images_saved < cfg.num_images_fid_score:
        current_batch = min(batch_size, cfg.num_images_fid_score - images_saved)
        
        gen_noise = torch.randn(current_batch, cfg.z_dim).to(cfg.device)
        
        with torch.inference_mode():
            # If generator is wrapped in DataParallel, use generator.module or handle normally
            generated_images = generator(gen_noise)
            
        for img in generated_images:
            # Convert Grayscale -> RGB to match the real images directory
            image_rgb = img.repeat(3, 1, 1)
            filename = os.path.join(generated_images_dir, f"generated_image_{images_saved:04d}.png")
            save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))
            images_saved += 1

    print(f"Successfully generated and saved {images_saved} images to {generated_images_dir}/")


def calculate_fid_sample(gen, loader, fid_metric):
    """
    Calculates FID score by comparing real images from the loader 
    with generated images from the generator.
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
                real_batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_batch, _ = next(data_iter)
                
            real_batch = real_batch[:batch_size].to(cfg.device)
           
            # Map [-1, 1] -> [0, 1] and expand grayscale to 3 channels
            real_rgb = (real_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0

            fid_metric.update(real_rgb, real=True)

            # --- 2. Process Fake Images ---
            # --- Fake Images ---
            noise = torch.randn(batch_size, cfg.z_dim, device=cfg.device)
            fake_batch = gen(noise)
            fake_rgb = (fake_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0
            
            fid_metric.update(fake_rgb, real=False)

    # --- 3. Compute and Log ---
    fid_score = fid_metric.compute().item()

    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    gen.train()
    return fid_score

def save_model(gen, disc, opt_gen, opt_disc, epoch, filename="checkpoint.pth"):
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    os.makedirs(gan_checkpoints_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_G_state_dict": opt_gen.state_dict(),
        "optimizer_D_state_dict": opt_disc.state_dict(),
    }
    
    save_path = f"{gan_checkpoints_dir}/{filename}"
    torch.save(checkpoint, save_path)
    print(f"--- Saved checkpoint: {filename} at epoch {epoch} ---")

def reload_checkpoint_model(gen, disc, opt_gen, opt_disc):
    checkpoint_path = f"{cfg.MODELS_DIR}/gan_checkpoints/latest_gan.pth"

    # 2. Check if a background run left a file behind
    if os.path.exists(checkpoint_path):
        print(" Found previous run checkpoint. Loading metadata...")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

        # Load the neural network and optimizer states
        gen.load_state_dict(checkpoint["generator_state_dict"])
        disc.load_state_dict(checkpoint["discriminator_state_dict"])
        opt_gen.load_state_dict(checkpoint["optimizer_G_state_dict"])
        opt_disc.load_state_dict(checkpoint["optimizer_D_state_dict"])

        # Start at the NEXT epoch (current saved epoch + 1)
        start_epoch = checkpoint["epoch"] + 1
        return start_epoch
    return 0


def load_best_model(gen, checkpoint_path):
    """
    Load the best model generator state dict from the given checkpoint path into the provided generator.
    If loading fails, prints an error and leaves the generator unchanged.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        gen.load_state_dict(checkpoint["generator_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    except Exception as e:
        print(f"Failed to load best model from {checkpoint_path}: {e}")
        # Fallback to just using current model
        pass


def log_tensorboard_visuals(wandb, gen, real_batch, gen_input, epoch):
    """
    Captures the current state of generation vs real images.
    Works for both standard GANs (gen_input = fixed noise) and SRGAN (gen_input = low-res images).
    """
    gen.eval()
    with torch.inference_mode():
        # 1. Generate fakes (Shape: N, 1, H, W)
        fake = gen(gen_input).reshape(
            -1, cfg.num_channels, cfg.image_size, cfg.image_size
        )

        # 2. Reshape real data (Shape: N, 1, H, W)
        real = real_batch.reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)

        # 3. Convert both from 1-channel to 3-channel (RGB)
        # This is necessary so the grid looks consistent in all viewers
        fake_rgb = fake.expand(-1, 3, -1, -1)
        real_rgb = real.expand(-1, 3, -1, -1)

        # 4. Create grids using Torchvision's built-in normalization
        # normalize=True: shifts the range to [0, 1]
        # value_range=(-1, 1): tells the function our Tanh/Transform output is [-1, 1]
        img_grid_fake = torchvision.utils.make_grid(
            fake_rgb, nrow=8, normalize=True, value_range=(-1, 1)
        )
        img_grid_real = torchvision.utils.make_grid(
            real_rgb, nrow=8, normalize=True, value_range=(-1, 1)
        )
        # 5. Log to TensorBoard
        # writer.add_image("Images/Generated", img_grid_fake, global_step=epoch)
        # writer.add_image("Images/Real", img_grid_real, global_step=epoch)
        # --- Make grids ---

        # --- Log to WandB ---
        wandb.log(
            {
                "Generated Grid": wandb.Image(
                    img_grid_fake, caption=f"epoch_{epoch:03d}"
                ),
                "Real Grid": wandb.Image(img_grid_real, caption=f"epoch_{epoch:03d}"),
                "epoch": epoch,
            },
            commit=False,
        )
    gen.train()


