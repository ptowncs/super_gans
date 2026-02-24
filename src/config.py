import os
import yaml

# Detect if running in Colab
IN_COLAB = 'COLAB_GPU' in os.environ
repo = "super_gans"
if IN_COLAB:
    PROJECT_PATH = f"/content/{repo}"
    DRIVE_PATH = f"/content/drive/MyDrive/{repo}"
else:
    PROJECT_PATH = os.path.abspath(os.path.join(".."))
    DRIVE_PATH = PROJECT_PATH
print(PROJECT_PATH)
os.chdir(PROJECT_PATH)
print(DRIVE_PATH)

DATA_DIR = f"{DRIVE_PATH}/data"
MODELS_DIR = f"{DRIVE_PATH}/saved_models"
RESULTS_DIR = f"{DRIVE_PATH}/results"


with open(f"{PROJECT_PATH}/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
print(config)
print(config['repos']['dataset_handle'])
