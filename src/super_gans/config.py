import os
from datetime import datetime
import torch
import yaml
from importlib import resources
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


#Detect if running in Kaggle
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Detect if running in Colab
IN_COLAB = 'COLAB_GPU' in os.environ
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

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
DATA_DIR = f"{DRIVE_PATH}/data"
MODELS_DIR = f"{DRIVE_PATH}/saved_models/{timestamp}"
RESULTS_DIR = f"{DRIVE_PATH}/results/{timestamp}"

with resources.files("super_gans").joinpath("config.yaml").open("r") as f:
    config = yaml.safe_load(f)
print(config)
print(config['repos']['dataset_handle'])

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-4
z_dim = 128
image_size = 128
num_channels = 1
num_workers = 4
image_dim = image_size * image_size * num_channels
batch_size = 64
num_epochs = 2000
classify_num_epochs = 15
classify_image_size = 224
num_images_fid_sample = 4 * batch_size #256
num_images_fid_score = 5000
fid_interval = 10
fid_dims = 2048
betas = (0.5, 0.999)

HIGH_RES = 128
LOW_RES = HIGH_RES // 4

# Change inside config.py:
high_res_transform = A.Compose([
    #not needed as done in both_transforms first
    #A.Resize(width=HIGH_RES, height=HIGH_RES),
    A.Normalize(mean=[0.5], std=[0.5]),  # One value for single-channel grayscale
    ToTensorV2(),
])

low_res_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=[0.5], std=[0.5]),   # One value for single-channel grayscale
    ToTensorV2(),
])

both_transforms = A.Compose(
    [
        # 1. First, resize the whole image safely to your 128x128 square canvas
        A.Resize(width=HIGH_RES, height=HIGH_RES), 
        # REMOVED A.RandomCrop completely to preserve global lung anatomy
        #A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        
        # 2. Mirror the image horizontally. 
        # (Safe because it just simulates looking at the X-ray from back-to-front)
        A.HorizontalFlip(p=0.5),
    
        # REMOVED A.RandomRotate90 completely!
        #A.RandomRotate90(p=0.5)
        # 3. Apply a subtle medical rotation instead of 90 degrees.
        # 'border_mode=cv2.BORDER_CONSTANT' ensures no weird mirroring artifacts on the edges.
        A.ShiftScaleRotate(
            shift_limit=0.05,    # Minor shifting (5% max)
            scale_limit=0.05,    # Minor zoom (5% max)
            rotate_limit=5,      # ONLY rotate up to 5 degrees! Prevents losing corners.
            border_mode=cv2.BORDER_CONSTANT, 
            value=0,             # Pads any tiny exposed edge with black
            p=0.5
        ),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)