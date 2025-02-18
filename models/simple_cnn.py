import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNModel(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNNModel, self).__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # (B, 3, H, W) -> (B, 16, H, W)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (B, 16, H, W) -> (B, 32, H, W)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B, 32, H, W) -> (B, 64, H, W)
        
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1)  # (B, 64, H, W) -> (B, n_classes, H, W)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # Shape: (B, 64, H, W)

        x = self.conv_out(x)  # (B, n_classes, H, W) - raw logits

        x = torch.argmax(x, dim=1)  # (B, H, W) - class index per pixel

        return x.float()  # Convert to float for compatibility if needed

# Example usage
model = SimpleCNNModel(n_classes=5)  # 5 classes
image = torch.randn(1, 3, 32, 32)  # Batch of 1, RGB image of size 32x32
mask = model(image)  # Output: (1, 102
