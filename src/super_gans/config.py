import os
from datetime import datetime
import torch
import yaml
from importlib import resources
from pathlib import Path

# Detect if running in Colab
IN_COLAB = 'COLAB_GPU' in os.environ
repo = "super_gans"
if IN_COLAB:
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
lr = 2e-4
z_dim = 64
image_size = 64
num_channels = 1
image_dim = image_size * image_size * num_channels
batch_size = 64
num_epochs = 3
num_real_images_to_save = 10
num_images_to_generate = 10
