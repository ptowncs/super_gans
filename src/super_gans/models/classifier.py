import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchmetrics import Accuracy, F1Score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import super_gans.config as cfg
from super_gans.utils import get_dataset_path
from collections import Counter
import wandb

class PneumoniaClassifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        # Use pretrained ResNet18 backbone
        self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Adjust first conv layer if input is 1 channel (if needed)
        # Note: Standard ResNet expects 3 channels. 
        # If your X-rays are grayscale, either repeat channels or use this:
        if self.model.conv1.in_channels != 3:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Replace classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_data(data_path):
    # Randomly flipping and rotating increases accuracy and avoids overfitting.
    # Training transforms include augmentation to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)), #ResNe4t models generally use 224x224
        transforms.RandomRotation(20),       # Randomly rotate by up to 20 degrees
        transforms.RandomHorizontalFlip(),   # Randomly Flip images horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        #Resnet18 models are trained on ImageNet dataset with these data distributions
    ])

    def is_valid_image(filename):
            return not filename.startswith("._")  # skip hidden macOS files

    dataset_path = get_dataset_path(data_path)
    chest_xray_ds =  f"{dataset_path}/chest_xray"
    train_dataset = datasets.ImageFolder(
        root=f"{chest_xray_ds}/train",
        transform=train_transforms,
        is_valid_file=is_valid_image,
    )

    # Validation/Test transforms only resize and normalize
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(
        root=f"{chest_xray_ds}/val",
        transform=val_transforms,
        is_valid_file=is_valid_image,)
    return train_dataset, val_dataset

def train_loop(train_dataset, val_dataset):
     # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    print(f"Classes found: {train_dataset.classes}")
    print(f"Class mapping: {train_dataset.class_to_idx}") # {'NORMAL': 0, 'PNEUMONIA': 1}
    print(f"Training samples: {len(train_dataset)}")
    
    counts = Counter(train_dataset.targets)

    # Sort by index to ensure [0, 1] order
    class_counts = [counts[i] for i in range(len(train_dataset.classes))]
    total_samples = sum(class_counts)

    # Calculate weights: Higher weight for fewer samples
    # Formula: weight = total_samples / (num_classes * class_count)
    weights = [total_samples / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(device)

    print(f"Samples per class: {class_counts}")
    print(f"Calculated weights: {class_weights}")


    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaClassifier(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights) #add class weights to overcome data imbalance
    # Metrics
    train_acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    val_acc_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    val_f1_metric = F1Score(task="multiclass", num_classes=2).to(device)
    
    for epoch in range(cfg.classify_num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_acc_metric.update(logits, y)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                val_loss += criterion(logits, y).item()
                val_acc_metric.update(logits, y)
                val_f1_metric.update(logits, y)

        # Compute metrics
        metrics = {
            "train/loss": train_loss / len(train_loader),
            "train/acc": train_acc_metric.compute(),
            "val/loss": val_loss / len(val_loader),
            "val/acc": val_acc_metric.compute(),
            "val/f1": val_f1_metric.compute(),
            "epoch": epoch
        }
        wandb.log(metrics)
        print(f"Epoch {epoch} logged to WandB. Val Acc: {metrics['val/acc']:.4f}")

        # Reset metrics for next epoch
        train_acc_metric.reset()
        val_acc_metric.reset()
        val_f1_metric.reset()
    return model

def createWandB():
    wandb.init(
        project="super-gans-project",
        name="pneumonia-resnet18-classification",
        config={
            "architecture": "ResNet18",
            "dataset": "Chest-XRay",
            "epochs": cfg.classify_num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "image_size": cfg.classify_image_size,
            })
    return wandb

def reload_model_eval():
    model = PneumoniaClassifier(num_classes=2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Use your specific optimizer

    checkpoint = torch.load('pneumonia_resnet18.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    model.eval()
    return model

def save_model(model, filename):
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    os.makedirs(gan_checkpoints_dir, exist_ok=True)

    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    }
    
    save_path = f"{gan_checkpoints_dir}/{filename}"
    torch.save(checkpoint, save_path)
    print(f"--- Saved checkpoint: {filename} ---")

def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("pneumonia_resnet18_model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/pneumonia_resnet18.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)

if __name__ == '__main__':
    wandb = createWandB()
    train_dataset, val_dataset = load_data("paultimothymooney/chest-xray-pneumonia/versions/2")
    model = train_loop(train_dataset, val_dataset)
    torch.save(model.state_dict(), 'pneumonia_resnet18.pth')
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    }
    save_model(model, 'pneumonia_resnet18.pth')
    uploadLogsAndMetricsToWandB(wandb)
    wandb.finish()