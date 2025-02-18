import os
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

from models.unet_parts import *
from models.simple_cnn import *

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")
if device == 'cuda': torch.cuda.empty_cache() # Clear the GPU memory cache


# Define dataset
class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[:100]  # Limit to 100 images
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])[:100]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load image (Assuming 3-channel PNG)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Convert to tensor (float16)
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Channels first
        mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Ensure integer mask for loss function
        
        if self.transform_image:
            image_tensor = self.transform_image(image_tensor)
        if self.transform_mask:
            mask_tensor = self.transform_mask(mask_tensor.unsqueeze(0)).squeeze(0) # squeeze as transform expects 3 dimensions

        return image_tensor, mask_tensor

# Paths
image_train_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/images" # all val for now
mask_train_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/masks"
image_val_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/images"
mask_val_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/masks"

# Dataset & DataLoader

# Perform transformations
transform_image = v2.Compose([
    v2.Resize((300, 300), antialias=True) # resize images
    ])
transform_mask = v2.Compose([
    v2.Resize((300, 300), antialias=True) # resize masks
    ])

train_data = COCOSegmentationDataset(image_train_dir, mask_train_dir, transform_image=transform_image, transform_mask=transform_mask)
val_data = COCOSegmentationDataset(image_val_dir, mask_val_dir, transform_image=transform_image, transform_mask=transform_mask)


batch_size = 2
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# UNet Model
# model = UNet(n_channels=3, n_classes=91).to(device)
# Simple CNN Model
model = SimpleCNNModel(n_classes=91).to(device)

# Loss & Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(dataloader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)  # Output shape: (B, 91, H, W)
        loss = criterion(pred, y)

        total_loss += loss.item()
        if batch % 10 == 0:
            logging.info(f"Batch {batch}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


# Validation function
def validate(dataloader, model, criterion):
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # Ensure y is long

            pred = model(X)  # Output shape: (B, 91, H, W)
            loss = criterion(pred, y)
            total_loss += loss.item()

            correct_pixels += (pred == y).sum().item()
            total_pixels += y.numel()

    accuracy = correct_pixels / total_pixels
    avg_loss = total_loss / len(dataloader)

    logging.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")
    return avg_loss, accuracy


# Training loop
epochs = 3
for epoch in range(epochs):
    logging.info(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(train_dataloader, model, criterion, optimizer)
    val_loss, val_acc = validate(val_dataloader, model, criterion)

# Save the model
model_path = "/home/maver02/Development/Models/COCO/instance_segmentation/unet_model_2.pth"
model_path = "/home/maver02/Development/Models/COCO/instance_segmentation/simple_cnn_model_2.pth"
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")
