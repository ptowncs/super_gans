import os
import gc       # Good practice for cleaning memory during training
import psutil   # Used for RAM tracking print statements
import kagglehub
import super_gans.config as cfg

# PyTorch Core & Dataset Handling
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, Subset
from torchvision.utils import save_image

# Evaluation Metrics (Inline Validation & Final Benchmarks)
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_fid import fid_score

def create_dirs():
    dirs = [cfg.DATA_DIR, cfg.RESULTS_DIR, cfg.MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_dataset_path(dataset_handle):
    # 1. (Optional) Force custom cache for Colab/Local
    # Note: Kaggle will ignore this and stay in /kaggle/input
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        os.environ['KAGGLEHUB_CACHE'] = cfg.DATA_DIR 
    
    # 2. Download and capture the environment-specific path
    # In Kaggle: returns /kaggle/input/...
    # In Colab: returns cfg.DATA_DIR/... or default cache
    dataset_path = kagglehub.dataset_download(dataset_handle)
    return dataset_path

def load_data(split="train"):
    def is_valid_image(filename):
        return not filename.startswith("._")  # skip hidden macOS files

    transforms_pipeline = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=cfg.num_channels
            ),  # Force 1 channel
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),  # image to tensor
            transforms.Normalize((0.5,), (0.5,)),  # normalize images , [0,1] to [-1,1]
        ]
    )
    dataset_path = get_dataset_path("paultimothymooney/chest-xray-pneumonia/versions/2")
    chest_xray_ds =  f"{dataset_path}/chest_xray"
    dataset = datasets.ImageFolder(
        root=f"{chest_xray_ds}/{split}",
        transform=transforms_pipeline,
        is_valid_file=is_valid_image,
    )
    return dataset

def build_fid_evaluation_dataset(target_samples=5000):
    val_ds = load_data(split="val")
    test_ds = load_data(split="test")
    
    eval_ds = ConcatDataset([val_ds, test_ds])
    current_count = len(eval_ds)
    print(f"Val + Test images: {current_count}")
    
    if current_count < target_samples:
        needed = target_samples - current_count
        print(f"Pulling the exact same {needed} sequential images from Train split...")
        
        train_ds = load_data(split="train")
        
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

def generate_images_fid(generator, generated_images_dir):
    # Generate random noise vectors
    gen_noise = torch.randn(cfg.num_images_fid_score, cfg.z_dim).to(cfg.device)

    # Generate images
    with torch.inference_mode():
        generated_images = generator(gen_noise).reshape(
            -1, cfg.num_channels, cfg.image_size, cfg.image_size
        )  # Reshape for saving/display

    os.makedirs(generated_images_dir, exist_ok=True)

    # Iterate through the batch and save each image separately
    for i, image in enumerate(generated_images):
        # Convert Grayscale -> RGB to match the real images directory
        image_rgb = image.repeat(3, 1, 1)
        filename = os.path.join(generated_images_dir, f"generated_image_{i:04d}.png")

        # Save the individual image (image tensor has shape (channels, height, width))
        save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))

    print(
        f"{cfg.num_images_fid_score} individual images saved to {generated_images_dir}/"
    )

def calc_fid_score(real_images_dir, generated_images_dir):
    # Paths to your image directories
    print(f"Path being checked:{repr(real_images_dir)}")
    if not os.path.exists(real_images_dir):
        raise ValueError(f"Real image path not found: {real_images_dir}")
    if not os.path.exists(generated_images_dir):
        raise ValueError(f"Generated image path not found: {generated_images_dir}")

    # (Make sure to populate `real_images` with your actual dataset)
    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size=50,  # Adjust batch size based on available GPU memory
        device = cfg.device,
        dims= cfg.fid_dims,  # Inception v3 output dimension
    )
    return fid_value

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
            real_rgb = (real_batch.expand(-1, 3, -1, -1) + 1.0) / 2.0

            fid_metric.update(real_rgb, real=True)

            # --- 2. Process Fake Images ---
            # --- Fake Images ---
            noise = torch.randn(batch_size, cfg.z_dim, device=cfg.device)
            fake_batch = gen(noise)
            fake_rgb = (fake_batch.expand(-1, 3, -1, -1) + 1.0) / 2.0
            
            fid_metric.update(fake_rgb, real=False)

    # --- 3. Compute and Log ---
    fid_score = fid_metric.compute().item()

    if cfg.device.type == 'cuda':
        torch.cuda.empty_cache()

    gen.train()
    return fid_score
