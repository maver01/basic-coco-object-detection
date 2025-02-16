import os
import cv2
import torch
import json
import time
import logging
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Create dataset and dataloader

class COCOSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing instance masks.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])[:100] # only consider first 100
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])[:100]

    def __len__(self):
        """
        Return the number of files in the image dataset (each image correspond to one mask)
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the images and masks
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name  # Image and mask have the same filename
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not mask_name:
            return None  # No mask found, handle accordingly
        
        # Set final image sizes (650x700), which includes all sizes
        image_size_h = 650
        image_size_w = 700

        # Load image
        image_original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # reading as is

        # Pad image to match desired size
        original_h, original_w, _ = image_original.shape
        pad_h = max(0, (image_size_h - original_h) // 2)
        pad_w = max(0, (image_size_w - original_w) // 2)

        image_padded = np.pad(image_original, ((pad_h, image_size_h - original_h - pad_h), (pad_w, image_size_w - original_w - pad_w), (0, 0)), mode='constant', constant_values=0)
        
        # Load mask (grayscale) and expand values
        mask_original = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Pad mask to match desired size instead of interpolation
        original_h, original_w = mask_original.shape

        pad_h = max(0, (image_size_h - original_h) // 2)
        pad_w = max(0, (image_size_w - original_w) // 2)

        mask_padded = np.pad(mask_original, ((pad_h, image_size_h - original_h - pad_h), (pad_w, image_size_w - original_w - pad_w)), mode='constant', constant_values=0)
        # Ensure mask values remain categorical (0 to 255 after expansion)
        mask_tensor = torch.tensor(mask_padded, dtype=torch.float32)
        # Convert image to tensor
        image_tensor = torch.tensor(image_padded, dtype=torch.float32).permute(2, 0, 1) # in tensors, channels must be first dimension
        return image_tensor, mask_tensor
    
image_val_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/images"
image_train_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/train/images"

mask_val_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/masks"
mask_train_dir = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/train/masks"

instances_val_dir = "/home/maver02/Development/Datasets/COCO/annotations/instances_val2017.json"
instances_train_dir = "/home/maver02/Development/Datasets/COCO/annotations/instances_val2017.json"

test_data = COCOSegmentationDataset(image_val_dir, mask_val_dir)
train_data = test_data # use test data for now as it is smaller

batch_size = 2  # Reduce to avoid OOM
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Model creation

# Select next item in dataloader
test_images, test_masks = next(iter(test_dataloader))
print(test_images.shape)
print(test_masks.shape, 'max value: ', test_masks.max())

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
logging.info(f'Using device {device}')

# Change here to adapt to your data
# n_channels=3 for RGB images
# n_classes is the number of probabilities you want to get per pixel

# Create model instance and move to device
model = UNet(n_channels=3, n_classes=91).to(device)

logging.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

print(model)


# Define loss, optimizer, epochs
loss_fn = nn.CrossEntropyLoss() # As we have multiclass represented as pixel integers in masks
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 3

## Train

def train(dataloader, model, loss_fn, optimizer):
    """
    In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
    and backpropagates the prediction error to adjust the model’s parameters.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.long()) # MS: might have to fix this. Only works when adding .long, but might need different processing

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """
    We also check the model’s performance against the test dataset to ensure it is learning.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")