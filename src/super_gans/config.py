import os
from datetime import datetime
import torch
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Detect if running in Kaggle
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Detect if running in Colab
IN_COLAB = "COLAB_GPU" in os.environ
print(f"IN_KAGGLE: {IN_KAGGLE}")
print(f"IN_COLAB: {IN_COLAB}")
repo = "super_gans"
if IN_KAGGLE:
    PROJECT_PATH = f"/kaggle/working/{repo}"
    DRIVE_PATH = f"/kaggle/working"
elif IN_COLAB:
    PROJECT_PATH = f"/content/{repo}"
    DRIVE_PATH = f"/content/drive/MyDrive/{repo}"
else:
    # Get the directory where this config.py file is located, then go up one level to project root
    current_file = Path(__file__).resolve()
    PROJECT_PATH = str(current_file.parents[2])
    DRIVE_PATH = PROJECT_PATH

print(f"Project Path: {PROJECT_PATH}")
print(f"Drive Path: {DRIVE_PATH}")

# Data directory in super_gans project
if IN_KAGGLE:
    # In Kaggle, use temporary directory to avoid read-only file system issues
    DATA_DIR = f"/tmp/super_gans/data"
else:
    DATA_DIR = f"{PROJECT_PATH}/data"
# Note: Individual functions should create their own subdirectories as needed

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# Directories for FID images (now under DATA_DIR for simplicity)
FID_REAL_DIR = f"{DATA_DIR}/fid_real"
FID_FAKE_DIR = f"{DATA_DIR}/fid_fake"
# Persistent directories for models and results (to keep after job completion)
# In Kaggle, avoid timestamp subdirs as the session is ephemeral (12-hour limit)
# In Colab/Local, keep timestamp subdirs for run organization
if IN_KAGGLE:
    MODELS_DIR = f"{DRIVE_PATH}/saved_models"
    RESULTS_DIR = f"{DRIVE_PATH}/results"
else:
    MODELS_DIR = f"{DRIVE_PATH}/saved_models/{timestamp}"
    RESULTS_DIR = f"{DRIVE_PATH}/results/{timestamp}"

# Data source selection: 'ptmooney' or 'rsna'
# In Kaggle environment, default to ptmooney unless explicitly overridden to rsna
# This prevents accidentally trying to load RSNA data in Kaggle without proper setup
if IN_KAGGLE:
    # In Kaggle, default to ptmooney dataset unless user explicitly wants rsna
    DATA_SOURCE = os.environ.get("DATA_SOURCE", "ptmooney").lower()
    if DATA_SOURCE not in ["ptmooney", "rsna"]:
        print(f"WARNING: Unknown DATA_SOURCE '{DATA_SOURCE}', defaulting to 'ptmooney'")
        DATA_SOURCE = "ptmooney"
elif IN_COLAB:
    # In Colab, require explicit setting
    DATA_SOURCE = os.environ.get("DATA_SOURCE", "ptmooney").lower()
    if DATA_SOURCE not in ["ptmooney", "rsna"]:
        print(f"WARNING: Unknown DATA_SOURCE '{DATA_SOURCE}', defaulting to 'ptmooney'")
        DATA_SOURCE = "ptmooney"
else:
    # Local environment
    DATA_SOURCE = os.environ.get("DATA_SOURCE", "ptmooney").lower()
    if DATA_SOURCE not in ["ptmooney", "rsna"]:
        print(f"WARNING: Unknown DATA_SOURCE '{DATA_SOURCE}', defaulting to 'ptmooney'")
        DATA_SOURCE = "ptmooney"

print(f"Using data source: {DATA_SOURCE}")

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = float(os.environ.get("LR", "1e-4"))  # learning rate
z_dim = int(os.environ.get("Z_DIM", "128"))
image_size = int(os.environ.get("IMAGE_SIZE", "128"))
num_channels = int(os.environ.get("NUM_CHANNELS", "1"))
num_workers = int(os.environ.get("NUM_WORKERS", "0"))
batch_size = int(os.environ.get("BATCH_SIZE", "64"))  # note: this was 64 in original config.py, not 32 from yaml
num_epochs = int(os.environ.get("NUM_EPOCHS", "2"))   # this was 2 in original config.py, not 400 from yaml
classify_num_epochs = int(os.environ.get("CLASSIFY_NUM_EPOCHS", "15"))
classify_image_size = int(os.environ.get("CLASSIFY_IMAGE_SIZE", "224"))
num_images_fid_sample = int(os.environ.get("NUM_IMAGES_FID_SAMPLE", str(4 * batch_size)))  # 256
num_images_fid_score = int(os.environ.get("NUM_IMAGES_FID_SCORE", "5000"))
fid_interval = int(os.environ.get("FID_INTERVAL", "1"))
fid_dims = int(os.environ.get("FID_DIMS", "2048"))
betas = (0.5, 0.999)

# WGAN / WGAN-GP specific hyperparameters - MADE CONFIGURABLE
n_critic = int(os.environ.get("N_CRITIC", "5"))  # Number of critic iterations per generator iteration
weight_clip = float(os.environ.get("WEIGHT_CLIP", "0.01"))    # Clipping parameter for original WGAN
lambda_gp = float(os.environ.get("LAMBDA_GP", "10"))        # Gradient penalty coefficient for WGAN-GP
# Alternative betas for WGAN/WGAN-GP (often beta1=0.0, beta2=0.9)
wgan_betas = (0.0, 0.9)

# Set high and low resolution - MADE CONFIGURABLE VIA ENVIRONMENT VARIABLES
HIGH_RES = int(os.environ.get("HIGH_RES", "128"))
LOW_RES = int(os.environ.get("LOW_RES", "32"))

# Validate that HIGH_RES is divisible by LOW_RES for SRGAN (if using SRGAN)
# Only validate if we're likely to use SRGAN - this affects all GANs but SRGAN has the 4x requirement
if HIGH_RES % LOW_RES != 0:
    print(f"WARNING: HIGH_RES ({HIGH_RES}) is not evenly divisible by LOW_RES ({LOW_RES}). "
          f"This may cause issues with SRGAN which expects integer division.")

# Change inside config.py:
high_res_transform = A.Compose(
    [
        # not needed as done in both_transforms first
        # A.Resize(width=HIGH_RES, height=HIGH_RES),
        A.Normalize(mean=[0.5], std=[0.5]),  # One value for single-channel grayscale
        ToTensorV2(),
    ]
)

low_res_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.5], std=[0.5]),  # One value for single-channel grayscale
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        # 1. First, resize the whole image safely to your 128x128 square canvas
        A.Resize(width=HIGH_RES, height=HIGH_RES),
        # REMOVED A.RandomCrop completely to preserve global lung anatomy
        # A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        # 2. Mirror the image horizontally.
        # (Safe because it just simulates looking at the X-ray from back-to-front)
        A.HorizontalFlip(p=0.5),
        # REMOVED A.RandomRotate90 completely!
        # A.RandomRotate90(p=0.5)
        # 3. Apply a subtle medical rotation instead of 90 degrees.
        # 'border_mode=cv2.BORDER_CONSTANT' ensures no weird mirroring artifacts on the edges.
        A.Affine(
            translate_percent=0.05,  # Minor shifting (5% max)
            scale=(0.95, 1.05),  # Minor zoom (5% max -> 0.95 to 1.05)
            rotate=5,  # ONLY rotate up to 5 degrees! Prevents losing corners.
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,  # Pads any tiny exposed edge with black
            p=0.5,
        ),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)


# Diffusion model parameters
diffusion_timesteps = int(os.environ.get("DIFFUSION_TIMESTEPS", "1000"))  # Number of diffusion steps
# We'll use a linear schedule from beta_start to beta_end over diffusion_timesteps
diffusion_beta_start = float(os.environ.get("DIFFUSION_BETA_START", "0.0001"))  # Starting value of beta schedule
diffusion_beta_end = float(os.environ.get("DIFFUSION_BETA_END", "0.02"))      # Ending value of beta schedule
