import json
import os
import random
import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PneumoniaKaggleDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]
        self.transform = transform
        if len(self.image_paths) == 0:
            raise ValueError("No Kaggle pneumonia images found!")
        print(f"Loaded {len(self.image_paths)} Kaggle images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, 1


class PneumoniaRsnaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [p for p in image_paths if p.suffix.lower() == ".dcm"]
        self.transform = transform
        if len(self.image_paths) == 0:
            raise ValueError("No RSNA pneumonia images found!")
        print(f"Loaded {len(self.image_paths)} RSNA images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = self._load_dicom_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 1

    def _load_dicom_image(self, dicom_path):
        dcm_data = pydicom.dcmread(str(dicom_path))
        img_arr = dcm_data.pixel_array

        if hasattr(dcm_data, "PhotometricInterpretation") and dcm_data.PhotometricInterpretation == "MONOCHROME1":
            img_arr = img_arr.max() - img_arr

        img_arr = img_arr.astype(np.float32)
        img_min = img_arr.min()
        img_max = img_arr.max()
        if img_max - img_min > 0:
            img_arr = (img_arr - img_min) / (img_max - img_min) * 255.0
        else:
            img_arr = np.zeros_like(img_arr)
        img_arr = img_arr.astype(np.uint8)

        return Image.fromarray(img_arr).convert("L")


def download_kaggle_dataset_to_dir(target_dir="./data/kaggle"):
    target_path = Path(target_dir)
    chest_xray_path = target_path / "chest_xray"

    if chest_xray_path.exists():
        print(f"Kaggle dataset exists at {chest_xray_path}")
        return chest_xray_path

    print("Downloading Kaggle pneumonia dataset...")
    dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    downloaded_chest_xray = Path(dataset_path) / "chest_xray"

    target_path.mkdir(parents=True, exist_ok=True)
    if downloaded_chest_xray.exists():
        shutil.copytree(downloaded_chest_xray, chest_xray_path)
        print(f"Copied Kaggle dataset to {chest_xray_path}")
    else:
        raise FileNotFoundError(f"Downloaded dataset not found at {downloaded_chest_xray}")

    return chest_xray_path


def get_kaggle_image_paths(data_dir="./data/kaggle"):
    data_path = Path(data_dir)
    chest_xray_path = data_path / "chest_xray"

    if not chest_xray_path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found at {chest_xray_path}")

    pneumonia_paths = []

    pneumonia_train_path = chest_xray_path / "train" / "PNEUMONIA"
    if pneumonia_train_path.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
            pneumonia_paths.extend(list(pneumonia_train_path.glob(ext)))
    else:
        raise FileNotFoundError(f"Could not find pneumonia images in Kaggle dataset at {pneumonia_train_path}")

    pneumonia_val_path = chest_xray_path / "val" / "PNEUMONIA"
    if pneumonia_val_path.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
            val_pneumonia_paths = list(pneumonia_val_path.glob(ext))
            pneumonia_paths.extend(val_pneumonia_paths)

    return pneumonia_paths


def save_data_split(train_paths, val_paths, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    train_paths_str = [str(p) for p in train_paths]
    val_paths_str = [str(p) for p in val_paths]

    split_data = {
        "train": train_paths_str,
        "validation": val_paths_str,
        "description": "Pneumonia dataset train/validation split",
        "total_images": len(train_paths_str) + len(val_paths_str),
        "train_count": len(train_paths_str),
        "validation_count": len(val_paths_str),
    }

    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=2)

    return output_file


def load_data_split(input_file):
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Split info file not found: {input_file}")

    with open(input_file, "r") as f:
        split_data = json.load(f)

    train_paths = [Path(p) for p in split_data["train"]]
    val_paths = [Path(p) for p in split_data["validation"]]

    return train_paths, val_paths


def create_kaggle_split_json(data_dir="./data", split_dir="./data/split"):
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    split_file = data_path / "kaggle_data_split.json"

    if split_file.exists():
        return load_data_split(split_file)

    all_paths = get_kaggle_image_paths(data_path / "kaggle")
    train_paths, val_paths = create_simple_train_val_split(all_paths, val_fraction=0.2)
    save_data_split(train_paths, val_paths, split_file)

    return train_paths, val_paths


def apply_kaggle_split(data_dir="./data", split_dir="./data/split"):
    split_dir = Path(split_dir)
    data_path = Path(data_dir)

    split_file = data_path / "kaggle_data_split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run create_kaggle_split_json first.")

    train_paths, val_paths = load_data_split(split_file)

    split_train_dir = split_dir / "train"
    split_val_dir = split_dir / "val"

    if split_train_dir.exists():
        shutil.rmtree(split_train_dir)
    split_train_dir.mkdir(parents=True)

    if split_val_dir.exists():
        shutil.rmtree(split_val_dir)
    split_val_dir.mkdir(parents=True)

    source_base = data_path / "kaggle" / "chest_xray"

    for src_path in train_paths:
        dst_path = split_train_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = split_train_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)

    for src_path in val_paths:
        dst_path = split_val_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = split_val_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)


def prepare_kaggle_data(data_dir="./data", split_dir="./data/split"):
    download_kaggle_dataset_to_dir(Path(data_dir) / "kaggle")
    create_kaggle_split_json(data_dir, split_dir)
    apply_kaggle_split(data_dir, split_dir)


def download_rsna_dataset_to_dir(target_dir="./data/rsna"):
    target_path = Path(target_dir)

    if target_path.exists() and any(target_path.rglob("*.dcm")):
        print(f"RSNA dataset exists at {target_path}")
        return target_path

    s3_url = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
    print(f"Downloading RSNA dataset from S3 mirror...")
    print(f"  Source: {s3_url}")

    home_dir = Path.home()
    cache_base = home_dir / ".cache" / "rsna_s3_fallback"
    cache_base.mkdir(parents=True, exist_ok=True)

    extract_dir = cache_base / "extraction"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    zip_path = extract_dir / "dataset.zip"
    print(f"  Downloading to: {zip_path}")
    import urllib.request
    urllib.request.urlretrieve(s3_url, zip_path)

    print(f"  Extracting dataset...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    items = [item for item in extract_dir.iterdir() if item.name != "dataset.zip"]
    if len(items) == 1 and items[0].is_dir():
        dataset_root = items[0]
    else:
        dataset_root = extract_dir

    dicom_files = list(dataset_root.rglob("*.dcm"))
    if not dicom_files:
        raise Exception("No DICOM files found after extraction")

    target_path.mkdir(parents=True, exist_ok=True)
    if dataset_root != target_path:
        shutil.copytree(dataset_root, target_path, dirs_exist_ok=True)

    return target_path


def get_rsna_image_paths(data_dir="./data/rsna"):
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"RSNA dataset not found at {data_path}")

    pneumonia_train_path = data_path / "train" / "PNEUMONIA"
    pneumonia_val_path = data_path / "val" / "PNEUMONIA"

    if pneumonia_train_path.exists() and pneumonia_val_path.exists():
        pneumonia_paths = []

        train_paths = list(pneumonia_train_path.rglob("*.dcm"))
        if train_paths:
            pneumonia_paths.extend(train_paths)

        val_paths = list(pneumonia_val_path.rglob("*.dcm"))
        if val_paths:
            pneumonia_paths.extend(val_paths)
    else:
        pneumonia_paths = list(data_path.rglob("*.dcm"))

    return pneumonia_paths


def create_rsna_split_json(data_dir="./data", split_dir="./data/split"):
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    split_file = data_path / "rsna_data_split.json"

    if split_file.exists():
        return load_data_split(split_file)

    all_paths = get_rsna_image_paths(data_path / "rsna")
    train_paths, val_paths = create_simple_train_val_split(all_paths, val_fraction=0.2)
    save_data_split(train_paths, val_paths, split_file)

    return train_paths, val_paths


def apply_rsna_split(data_dir="./data", split_dir="./data/split"):
    split_dir = Path(split_dir)
    data_path = Path(data_dir)

    split_file = data_path / "rsna_data_split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run create_rsna_split_json first.")

    train_paths, val_paths = load_data_split(split_file)

    split_train_dir = split_dir / "train"
    split_val_dir = split_dir / "val"

    if split_train_dir.exists():
        shutil.rmtree(split_train_dir)
    split_train_dir.mkdir(parents=True)

    if split_val_dir.exists():
        shutil.rmtree(split_val_dir)
    split_val_dir.mkdir(parents=True)

    source_base = data_path / "rsna"

    for src_path in train_paths:
        dst_path = split_train_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = split_train_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)

    for src_path in val_paths:
        dst_path = split_val_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = split_val_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)


def prepare_rsna_data(data_dir="./data", split_dir="./data/split"):
    download_rsna_dataset_to_dir(Path(data_dir) / "rsna")
    create_rsna_split_json(data_dir, split_dir)
    apply_rsna_split(data_dir, split_dir)


def clean_data_splits(data_dir="./data", split_dir="./data/split"):
    data_path = Path(data_dir)
    split_path = Path(split_dir)

    kaggle_path = data_path / "kaggle"
    rsna_path = data_path / "rsna"

    if kaggle_path.exists():
        shutil.rmtree(kaggle_path)

    if rsna_path.exists():
        shutil.rmtree(rsna_path)

    split_train_path = split_path / "train"
    split_val_path = split_path / "val"

    if split_train_path.exists():
        shutil.rmtree(split_train_path)

    if split_val_path.exists():
        shutil.rmtree(split_val_path)

    split_path.mkdir(parents=True, exist_ok=True)


def create_simple_train_val_split(image_paths, val_fraction=0.2, shuffle=True):
    if shuffle:
        random.seed(42)
        shuffled_paths = image_paths.copy()
        random.shuffle(shuffled_paths)
    else:
        shuffled_paths = image_paths.copy()

    split_idx = int(len(shuffled_paths) * (1.0 - val_fraction))
    train_paths = shuffled_paths[:split_idx]
    val_paths = shuffled_paths[split_idx:]

    return train_paths, val_paths