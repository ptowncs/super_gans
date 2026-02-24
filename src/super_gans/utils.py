import os
import kagglehub
import super_gans.config as cfg

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