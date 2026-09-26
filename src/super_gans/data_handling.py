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
import csv  # Added for CSV handling (using built-in csv module)

import super_gans.config as cfg


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
        # All Kaggle images are pneumonia (label 1)
        return image, 1  # All Kaggle images are pneumonia (label 1)


class PneumoniaRsnaDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        """
        Args:
            image_paths: List of Path objects to DICOM files
            labels: Optional list of integer labels (0=normal, 1=pneumonia)
                   If None, labels will be inferred from directory structure:
                   - Images in directories containing 'normal' (case-insensitive) -> label 0
                   - Images in directories containing 'pneumonia' (case-insensitive) -> label 1
                   - If neither pattern matches, defaults to label 1 (pneumonia) for backward compatibility
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = [p for p in image_paths if p.suffix.lower() == ".dcm"]
        self.transform = transform

        if len(self.image_paths) == 0:
            raise ValueError("No RSNA images found!")

        # If labels provided explicitly, use them
        if labels is not None:
            self.labels = labels
            # Verify we have labels for all images
            if len(self.labels) != len(self.image_paths):
                raise ValueError(f"Number of labels ({len(self.labels)}) does not match number of images ({len(self.image_paths)})")

            normal_count = sum(1 for l in self.labels if l == 0)
            pneumonia_count = sum(1 for l in self.labels if l == 1)
            print(f"Loaded {len(self.image_paths)} RSNA images ({normal_count} normal, {pneumonia_count} pneumonia) [explicit labels]")
        else:
            # Infer labels from directory structure
            self.labels = []
            normal_count = 0
            pneumonia_count = 0
            default_count = 0

            for img_path in self.image_paths:
                # Check if any part of the path contains normal/pneumonia indicators
                path_str = str(img_path).lower()
                if 'normal' in path_str:
                    self.labels.append(0)
                    normal_count += 1
                elif 'pneumonia' in path_str:
                    self.labels.append(1)
                    pneumonia_count += 1
                else:
                    # Default to pneumonia for backward compatibility
                    self.labels.append(1)
                    default_count += 1
                    pneumonia_count += 1  # Count as pneumonia since we defaulted to 1

            print(f"Loaded {len(self.image_paths)} RSNA images "
                  f"({normal_count} normal, {pneumonia_count} pneumonia) [inferred from directories]"
                  + (f", {default_count} defaulted to pneumonia" if default_count > 0 else ""))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = self._load_dicom_image(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

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


def download_ptmooney_dataset_to_dir(target_dir="./data/ptmooney"):
    """
    Download the Kaggle pneumoniadata (paultimothymooney/chest-xray-pneumonia).
    This dataset contains chest X-ray images organized in train/val/test folders
    with NORMAL and PNEUMONIA subdirectories.
    """
    target_path = Path(target_dir)
    chest_xray_path = target_path / "chest_xray"

    if chest_xray_path.exists():
        print(f"PTMOONEY dataset exists at {chest_xray_path}")
        return chest_xray_path

    print("Downloading PTMOONEY Kaggle pneumonia dataset...")
    dataset_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    downloaded_chest_xray = Path(dataset_path) / "chest_xray"

    target_path.mkdir(parents=True, exist_ok=True)
    if downloaded_chest_xray.exists():
        shutil.copytree(downloaded_chest_xray, chest_xray_path)
        print(f"Copied PTMOONEY dataset to {chest_xray_path}")
    else:
        raise FileNotFoundError(f"Downloaded dataset not found at {downloaded_chest_xray}")

    return chest_xray_path


def download_rsna_data_and_labels(data_dir=str(Path(cfg.DATA_DIR) / "rsna"), label_dir=str(Path(cfg.DATA_DIR) / "rsna_labels")):
    """
    Download RSNA pneumonia dataset and labels using kagglehub competition download.
    Simple unified approach that gets both images and labels in one call.
    """
    import kagglehub

    # Set up expected paths
    data_path = Path(data_dir)
    label_path = Path(label_dir)
    stage_2_train_path = data_path / "stage_2_train_images"
    stage2_label_path = label_path / "stage_2_train_labels.csv"

    # Check if we already have what we need
    if stage_2_train_path.exists() and any(stage_2_train_path.iterdir()) and stage2_label_path.exists():
        print(f"RSNA dataset and labels already exist at {data_path} and {label_path}")
        return data_path, label_path

    print("Downloading RSNA Pneumonia Detection Challenge dataset and labels via kagglehub competition download...")

    try:
        # Download the entire competition (images + labels)
        competition_path = kagglehub.competition_download('rsna-pneumonia-detection-challenge')
        competition_path = Path(competition_path)
        print(f"Downloaded competition files to: {competition_path}")

        # Organize the files to expected locations
        # 1. Handle images: copy stage_2_train_images
        downloaded_stage_2_images = competition_path / "stage_2_train_images"
        if downloaded_stage_2_images.exists():
            data_path.mkdir(parents=True, exist_ok=True)

            # Remove existing if present
            if stage_2_train_path.exists():
                shutil.rmtree(stage_2_train_path)
            shutil.copytree(downloaded_stage_2_images, stage_2_train_path)
            print(f"Copied stage_2_train_images to {stage_2_train_path}")
        else:
            # Search for it if not at root
            found_stage_2 = list(competition_path.rglob("stage_2_train_images"))
            if found_stage_2:
                downloaded_stage_2_images = found_stage_2[0]
                data_path.mkdir(parents=True, exist_ok=True)
                if stage_2_train_path.exists():
                    shutil.rmtree(stage_2_train_path)
                shutil.copytree(downloaded_stage_2_images, stage_2_train_path)
                print(f"Copied stage_2_train_images from {downloaded_stage_2_images} to {stage_2_train_path}")
            else:
                raise Exception("Could not find stage_2_train_images in downloaded competition")

        # 2. Handle labels: copy stage_2_train_labels.csv
        downloaded_stage_2_labels = competition_path / "stage_2_train_labels.csv"
        if downloaded_stage_2_labels.exists():
            label_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_stage_2_labels, label_path / "stage_2_train_labels.csv")
            print(f"Copied stage_2_train_labels.csv to {label_path}")
        else:
            # Search for it if not at root
            found_labels = list(competition_path.rglob("stage_2_train_labels.csv"))
            if found_labels:
                downloaded_stage_2_labels = found_labels[0]
                label_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_stage_2_labels, label_path / "stage_2_train_labels.csv")
                print(f"Copied stage_2_train_labels.csv from {downloaded_stage_2_labels} to {label_path}")
            else:
                raise Exception("Could not find stage_2_train_labels.csv in downloaded competition")

        print(f"Successfully organized RSNA dataset and labels:")
        print(f"  Images: {stage_2_train_path}")
        print(f"  Labels: {label_path}")
        return data_path, label_path

    except Exception as e:
        print(f"Failed to download RSNA competition data: {e}")
        print("\nPlease download manually from:")
        print("https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data")
        print("Then ensure you have:")
        print(f"  - {stage_2_train_path}/ (DICOM images)")
        print(f"  - {label_path}/stage_2_train_labels.csv")
        raise Exception(f"Failed to download RSNA competition data: {e}")


def get_ptmooney_image_paths(data_dir=str(Path(cfg.DATA_DIR) / "ptmooney")):
    """
    Get PTMOONEY Kaggle image paths (paultimothymooney/chest-xray-pneumonia).
    This dataset contains only pneumonia images (label=1) organized in train/val/test folders.
    """
    data_path = Path(data_dir)
    chest_xray_path = data_path / "chest_xray"

    if not chest_xray_path.exists():
        raise FileNotFoundError(f"PTMOONEY dataset not found at {chest_xray_path}")

    pneumonia_paths = []

    pneumonia_train_path = chest_xray_path / "train" / "PNEUMONIA"
    if pneumonia_train_path.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", ".tiff"]:
            pneumonia_paths.extend(list(pneumonia_train_path.glob(ext)))
    else:
        raise FileNotFoundError(f"Could not find pneumonia images in PTMOONEY dataset at {pneumonia_train_path}")

    pneumonia_val_path = chest_xray_path / "val" / "PNEUMONIA"
    if pneumonia_val_path.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", ".tiff"]:
            val_pneumonia_paths = list(pneumonia_val_path.glob(ext))
            pneumonia_paths.extend(val_pneumonia_paths)

    return pneumonia_paths


def get_rsna_image_paths_and_labels(data_dir=str(Path(cfg.DATA_DIR) / "rsna"), label_dir=str(Path(cfg.DATA_DIR) / "rsna_labels"), label_filter=None):
    """
    Get RSNA image paths and their corresponding labels.
    Args:
        data_dir: Path to RSNA dataset directory
        label_dir: Path to directory containing label CSV files
        label_filter: Optional list of labels to include (e.g., [1] for pneumonia-only, [0] for normal-only, [0,1] for all)
                     If None, returns all labeled images
    Returns:
        If label_filter is None: tuples of (image_paths, labels) where labels are 0 (normal) or 1 (pneumonia)
        If label_filter is provided: list of image_paths only (for filtered datasets)
    """
    data_path = Path(data_dir)
    label_path = Path(label_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"RSNA dataset not found at {data_path}")

    # Download labels if needed (using the combined download function which handles both images and labels)
    if not label_path.exists():
        download_rsna_data_and_labels(str(data_path), str(label_path))

    # Load the stage 2 training labels (primary labels)
    stage2_label_file = label_path / "stage_2_train_labels.csv"
    if not stage2_label_file.exists():
        raise FileNotFoundError(f"RSNA label file not found at {stage2_label_file}. "
                              "Please ensure label files are downloaded or specify correct label_dir.")

    # Load labels CSV using built-in csv module and normalize patient IDs for matching
    patient_to_label = {}
    try:
        with open(stage2_label_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['patientId']
                target = int(row['Target'])
                # Normalize patient ID for consistent matching (strip whitespace)
                normalized_patient_id = str(patient_id).strip()
                patient_to_label[normalized_patient_id] = target  # 0=normal, 1=pneumonia
    except Exception as e:
        raise RuntimeError(f"Error reading RSNA label CSV file {stage2_label_file}: {e}")

    print(f"Loaded labels for {len(patient_to_label)} unique patients from RSNA labels")

    # Find all DICOM files and extract patientId from DICOM metadata
    all_dicom_paths = list(data_path.rglob("*.dcm"))

    image_paths = []
    labels = []

    for dicom_path in all_dicom_paths:
        try:
            # Read DICOM to get PatientID from metadata
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            patient_id = getattr(ds, 'PatientID', None)

            # Skip if no PatientID found
            if patient_id is None:
                continue

            # Normalize patient ID for comparison (strip whitespace)
            normalized_patient_id = str(patient_id).strip() if patient_id is not None else None

            # Look up label for this patientId using normalized ID
            if normalized_patient_id is not None and normalized_patient_id in patient_to_label:
                image_paths.append(dicom_path)
                labels.append(patient_to_label[normalized_patient_id])
            # Note: If no matching label found, we skip the image
            # This ensures we only use images with known labels
        except Exception as e:
            # Skip unreadable DICOM files
            print(f"Warning: Could not read DICOM file {dicom_path}: {e}")
            continue

    print(f"Found {len(image_paths)} DICOM images with known labels out of {len(all_dicom_paths)} total DICOM files")

    if len(image_paths) == 0:
        raise ValueError("No RSNA images found with matching labels!")

    # Apply label filter if specified
    if label_filter is not None:
        filtered_paths = []
        for img_path, label in zip(image_paths, labels):
            if label in label_filter:
                filtered_paths.append(img_path)

        # Log filtering results
        original_normal = sum(1 for l in labels if l == 0)
        original_pneumonia = sum(1 for l in labels if l == 1)

        if label_filter == [0]:
            filtered_normal, filtered_pneumonia = len(filtered_paths), 0
            print(f"After filtering for normal images only (label=0): "
                  f"{filtered_normal} normal, {filtered_pneumonia} pneumonia "
                  f"(from {original_normal} normal, {original_pneumonia} pneumonia)")
        elif label_filter == [1]:
            filtered_normal, filtered_pneumonia = 0, len(filtered_paths)
            print(f"After filtering for pneumonia images only (label=1): "
                  f"{filtered_normal} normal, {filtered_pneumonia} pneumonia "
                  f"(from {original_normal} normal, {original_pneumonia} pneumonia)")
        elif label_filter == [0, 1]:
            filtered_normal, filtered_pneumonia = original_normal, original_pneumonia
            print(f"No filtering applied (keeping all labels {label_filter}): "
                  f"{filtered_normal} normal, {filtered_pneumonia} pneumonia "
                  f"(from {original_normal} normal, {original_pneumonia} pneumonia)")
        else:
            # Custom filter - count actual labels in filtered set
            # This is less common but we'll handle it correctly
            filtered_labels = []
            for img_path in filtered_paths:
                # Need to get the label for this image path
                relative_path = img_path.relative_to(data_path)
                patient_id = str(relative_path.parts[0])
                if patient_id in patient_to_label:
                    filtered_labels.append(patient_to_label[patient_id])

            filtered_normal = sum(1 for l in filtered_labels if l == 0)
            filtered_pneumonia = sum(1 for l in filtered_labels if l == 1)
            print(f"After filtering for labels {label_filter}: "
                  f"{filtered_normal} normal, {filtered_pneumonia} pneumonia "
                  f"(from {original_normal} normal, {original_pneumonia} pneumonia)")

        return filtered_paths

    # No filter - return paths and labels
    # Log label distribution
    normal_count = sum(1 for l in labels if l == 0)
    pneumonia_count = sum(1 for l in labels if l == 1)
    print(f"Label distribution: {normal_count} normal, {pneumonia_count} pneumonia")

    return image_paths, labels


def save_data_split(train_paths, val_paths, output_file, base_dir=None):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if base_dir is not None:
        base_path = Path(base_dir)
        # Convert absolute paths to relative paths based on base_dir
        train_paths_str = []
        val_paths_str = []
        for p in train_paths:
            try:
                # If path is absolute and within base_dir, make it relative
                if p.is_absolute() and p.resolve().is_relative_to(base_path.resolve()):
                    train_paths_str.append(str(p.relative_to(base_path)))
                else:
                    train_paths_str.append(str(p))
            except ValueError:
                # If relative_to fails, keep the original path
                train_paths_str.append(str(p))
        for p in val_paths:
            try:
                # If path is absolute and within base_dir, make it relative
                if p.is_absolute() and p.resolve().is_relative_to(base_path.resolve()):
                    val_paths_str.append(str(p.relative_to(base_path)))
                else:
                    val_paths_str.append(str(p))
            except ValueError:
                # If relative_to fails, keep the original path
                val_paths_str.append(str(p))
    else:
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


def load_data_split(input_file, base_dir=None):
    # If input_file is just a filename, look for it in the module directory
    input_path = Path(input_file)
    if not input_path.is_absolute() and not input_path.parent.parts:
        # Looks like just a filename, look in module directory
        module_dir = Path(__file__).resolve().parent
        input_path = module_dir / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Split info file not found: {input_path}")

    with open(input_path, "r") as f:
        split_data = json.load(f)

    if base_dir is not None:
        base_path = Path(base_dir)
        # Convert relative paths to absolute paths based on base_dir
        train_paths = []
        val_paths = []
        for p in split_data["train"]:
            p_obj = Path(p)
            if not p_obj.is_absolute():
                train_paths.append(base_path / p)
            else:
                train_paths.append(p_obj)
        for p in split_data["validation"]:
            p_obj = Path(p)
            if not p_obj.is_absolute():
                val_paths.append(base_path / p)
            else:
                val_paths.append(p_obj)
    else:
        train_paths = [Path(p) for p in split_data["train"]]
        val_paths = [Path(p) for p in split_data["validation"]]

    return train_paths, val_paths


def create_ptmooney_split_json(data_dir=str(Path(cfg.DATA_DIR) / "ptmooney"), split_dir=str(Path(cfg.DATA_DIR) / "split")):
    """
    Create train/validation split for PTMOONEY Kaggle dataset (paultimothymooney/chest-xray-pneumonia).
    """
    # Get the directory where this module is located for JSON persistence
    module_dir = Path(__file__).resolve().parent
    module_dir.mkdir(parents=True, exist_ok=True)

    # For data processing, use the passed-in directories
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Save/load JSON from module directory
    split_file = module_dir / "ptmooney_data_split.json"

    if split_file.exists():
        return load_data_split(split_file, base_dir=data_path)

    all_paths = get_ptmooney_image_paths(data_path)
    train_paths, val_paths = create_simple_train_val_split(all_paths, val_fraction=0.2)
    save_data_split(train_paths, val_paths, split_file, base_dir=data_path)

    return train_paths, val_paths


def create_rsna_split_json(data_dir=str(Path(cfg.DATA_DIR) / "rsna"), split_dir=str(Path(cfg.DATA_DIR) / "split"), label_dir=None):
    """
    Create train/validation split for RSNA dataset using image paths and labels
    """
    # If label_dir is None, set it to the default location
    if label_dir is None:
        label_dir = str(Path(data_dir) / "rsna_labels")

    # Get the directory where this module is located for JSON persistence
    module_dir = Path(__file__).resolve().parent
    module_dir.mkdir(parents=True, exist_ok=True)

    # For data processing, use the passed-in directories
    data_path = Path(data_dir)
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Save/load JSON from module directory
    split_file = module_dir / "rsna_data_split.json"

    if split_file.exists():
        # Load existing split and also load labels for compatibility
        train_paths, val_paths = load_data_split(split_file, base_dir=data_path)
        with open(split_file, "r") as f:
            split_data = json.load(f)
        train_labels = split_data.get("train_labels", [])
        val_labels = split_data.get("validation_labels", [])

        # If we don't have labels in the split file, try to infer from directory structure
        if not train_labels or not val_labels:
            # For backward compatibility, we need to infer labels from directory structure
            # or load them from the original source. For now, we'll extract from split directory
            train_labels = []
            val_labels = []

            # Try to infer labels from existing split directory structure
            split_train_dir = split_path / "train"
            split_val_dir = split_path / "val"

            if split_train_dir.exists():
                # Count files in pneumonia vs normal subdirectories
                pneumonia_train = list((split_train_dir / "pneumonia").glob("*"))
                normal_train = list((split_train_dir / "normal").glob("*"))
                train_labels = [1] * len(pneumonia_train) + [0] * len(normal_train)
                # Ensure we have the right number of labels
                if len(train_labels) != len(train_paths):
                    # Fallback: assume all are pneumonia (for backward compatibility)
                    train_labels = [1] * len(train_paths)

            if split_val_dir.exists():
                # Count files in pneumonia vs normal subdirectories
                pneumonia_val = list((split_val_dir / "pneumonia").glob("*"))
                normal_val = list((split_val_dir / "normal").glob("*"))
                val_labels = [1] * len(pneumonia_val) + [0] * len(normal_val)
                # Ensure we have the right number of labels
                if len(val_labels) != len(val_paths):
                    # Fallback: assume all are pneumonia (for backward compatibility)
                    val_labels = [1] * len(val_paths)

        return train_paths, val_paths, train_labels, val_labels

    # Get image paths and labels
    all_paths, all_labels = get_rsna_image_paths_and_labels(data_path, label_dir)

    # Validate that we found valid image-label pairs
    if len(all_paths) == 0:
        raise ValueError("No valid RSNA image-label pairs found. Please check that DICOM files contain valid PatientID values that match the label CSV.")

    # Create combined list of (path, label) tuples for stratified splitting
    path_label_pairs = list(zip(all_paths, all_labels))

    # Simple random split (we could make this stratified by label if desired)
    random.seed(42)  # For reproducibility
    random.shuffle(path_label_pairs)

    split_idx = int(len(path_label_pairs) * (1.0 - 0.2))  # 80% train, 20% val
    train_pairs = path_label_pairs[:split_idx]
    val_pairs = path_label_pairs[split_idx:]

    # Unzip back to separate lists
    train_paths, train_labels = zip(*train_pairs) if train_pairs else ([], [])
    val_paths, val_labels = zip(*val_pairs) if val_pairs else ([], [])

    # Convert to lists (zip returns tuples in Python 3)
    train_paths = list(train_paths)
    train_labels = list(train_labels) if train_labels else []
    val_paths = list(val_paths)
    val_labels = list(val_labels) if val_labels else []

    # Save split information including labels for verification
    split_data = {
        "train": [str(p) for p in train_paths],
        "validation": [str(p) for p in val_paths],
        "train_labels": train_labels,  # Store labels for verification/debugging
        "validation_labels": val_labels,
        "description": "RSNA pneumonia dataset train/validation split with labels",
        "total_images": len(train_paths) + len(val_paths),
        "train_count": len(train_paths),
        "validation_count": len(val_paths),
        "train_normal_count": sum(1 for l in train_labels if l == 0),
        "train_pneumonia_count": sum(1 for l in train_labels if l == 1),
        "val_normal_count": sum(1 for l in val_labels if l == 0),
        "val_pneumonia_count": sum(1 for l in val_labels if l == 1),
    }

    # Save the split file using save_data_split for relative paths
    save_data_split(train_paths, val_paths, split_file, base_dir=data_path)

    # Add labels to the split file for verification/debugging
    with open(split_file, "r") as f:
        split_data_with_labels = json.load(f)
    split_data_with_labels["train_labels"] = train_labels
    split_data_with_labels["validation_labels"] = val_labels
    split_data_with_labels["description"] = "RSNA pneumonia dataset train/validation split with labels"
    split_data_with_labels["train_normal_count"] = sum(1 for l in train_labels if l == 0)
    split_data_with_labels["train_pneumonia_count"] = sum(1 for l in train_labels if l == 1)
    split_data_with_labels["val_normal_count"] = sum(1 for l in val_labels if l == 0)
    split_data_with_labels["val_pneumonia_count"] = sum(1 for l in val_labels if l == 1)
    with open(split_file, "w") as f:
        json.dump(split_data_with_labels, f, indent=2)

    # Also save a human-readable summary
    summary_file = data_path / "rsna_split_summary.json"
    with open(summary_file, "w") as f:
        json.dump(split_data_with_labels, f, indent=2)

    return train_paths, val_paths, train_labels, val_labels


def apply_ptmooney_split(data_dir=str(Path(cfg.DATA_DIR) / "ptmooney"), split_dir=str(Path(cfg.DATA_DIR) / "split")):
    """
    Apply the PTMOONEY Kaggle train/validation split by copying files to split directories.
    """
    split_dir = Path(split_dir)
    data_path = Path(data_dir)

    # Get the directory where this module is located for JSON persistence
    module_dir = Path(__file__).resolve().parent
    split_file = module_dir / "ptmooney_data_split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}. Run create_ptmooney_split_json first.")

    train_paths, val_paths = load_data_split(split_file, base_dir=data_path)

    split_train_dir = split_dir / "train"
    split_val_dir = split_dir / "val"

    if split_train_dir.exists():
        shutil.rmtree(split_train_dir)
    split_train_dir.mkdir(parents=True)

    if split_val_dir.exists():
        shutil.rmtree(split_val_dir)
    split_val_dir.mkdir(parents=True)

    source_base = data_path / "chest_xray"

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


def apply_rsna_split(data_dir=str(Path(cfg.DATA_DIR) / "rsna"), split_dir=str(Path(cfg.DATA_DIR) / "split"), label_dir=None):
    """
    Apply the RSNA train/validation split by copying files to split directories
    """
    # If label_dir is None, set it to the default location
    if label_dir is None:
        label_dir = str(Path(data_dir) / "rsna_labels")

    split_dir = Path(split_dir)
    data_path = Path(data_dir)

    # Get the directory where this module is located for JSON persistence
    module_dir = Path(__file__).resolve().parent

    # Get the split with labels
    result = create_rsna_split_json(data_dir, split_dir, label_dir)
    train_paths, val_paths, train_labels, val_labels = result

    split_train_dir = split_dir / "train"
    split_val_dir = split_dir / "val"

    if split_train_dir.exists():
        shutil.rmtree(split_train_dir)
    split_train_dir.mkdir(parents=True)

    if split_val_dir.exists():
        shutil.rmtree(split_val_dir)
    split_val_dir.mkdir(parents=True)

    source_base = data_path

    # Copy training images
    for src_path, label in zip(train_paths, train_labels):
        # Create subdirectories for normal/pneumonia for clarity (optional)
        label_str = "pneumonia" if label == 1 else "normal"
        dst_dir = split_train_dir / label_str
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)

    # Copy validation images
    for src_path, label in zip(val_paths, val_labels):
        # Create subdirectories for normal/pneumonia for clarity (optional)
        label_str = "pneumonia" if label == 1 else "normal"
        dst_dir = split_val_dir / label_str
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / src_path.name
        counter = 1
        original_dst_path = dst_path
        while dst_path.exists():
            stem = original_dst_path.stem
            suffix = original_dst_path.suffix
            dst_path = dst_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src_path, dst_path)


def prepare_ptmooney_data(data_dir=str(Path(cfg.DATA_DIR) / "ptmooney"), split_dir=str(Path(cfg.DATA_DIR) / "split")):
    """
    Prepare the Kaggle pneumoniadata (paultimothymooney/chest-xray-pneumonia).
    Downloads images, creates train/validation split, and applies the split.
    This dataset contains only pneumonia images (label=1) organized in train/test/val folders.
    """
    download_ptmooney_dataset_to_dir(data_dir)
    create_ptmooney_split_json(data_dir, split_dir)
    apply_ptmooney_split(data_dir, split_dir)


def prepare_rsna_data(data_dir=str(Path(cfg.DATA_DIR) / "rsna"), split_dir=str(Path(cfg.DATA_DIR) / "split"), label_dir=None):
    """
    Prepare RSNA data: download images, download labels, create split, apply split
    Downloads label files if they don't exist locally.
    If label_dir is None, labels will be placed in data_dir/rsna_labels
    """
    # If label_dir not specified, place labels alongside images in data_dir
    if label_dir is None:
        label_dir = str(Path(data_dir) / "rsna_labels")

    # Download/verify RSNA images and labels exist
    rsna_data_path = Path(data_dir)
    downloaded_data_path, downloaded_label_path = download_rsna_data_and_labels(rsna_data_path, label_dir)

    # The images are already in the correct location after download_rsna_data_and_labels
    image_data_path = downloaded_data_path

    # Create and apply the split
    create_rsna_split_json(image_data_path, split_dir, downloaded_label_path)
    apply_rsna_split(image_data_path, split_dir, downloaded_label_path)


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