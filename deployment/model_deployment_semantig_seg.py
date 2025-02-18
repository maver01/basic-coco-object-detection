import torch
import cv2
import numpy as np
from torchvision.transforms import v2
from models.unet_parts import *
from models.simple_cnn import *


model_path = "/home/maver02/Development/Models/COCO/instance_segmentation/simple_cnn_model_2.pth"
image_path = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/images/000000000139.png"
mask_path = "/home/maver02/Development/Datasets/COCO/preprocess_coco_2_v1/val/masks/000000000139.png"


# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNNModel(n_classes=91).to(device) # initialize the model
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Load an image and its original mask
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Read as is (keep original depth)
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# Convert to tensor
image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) 

# Perform transformations
transform_image = v2.Compose([
    v2.Resize((300, 300), antialias=True) # resize images
    ])

image_tensor = transform_image(image_tensor).to(device)

# Run inference
with torch.no_grad():
    output = model(image_tensor.unsqueeze(0))

print(output.shape)

# Get the predicted segmentation mask
pred_mask = output.argmax(dim=1).squeeze().cpu().numpy() # Convert to NumPy array

pred_mask_viz = (pred_mask * 255).astype(np.uint8)

# Get original mask shape
original_size = (mask.shape[1], mask.shape[0])  # (width, height)

# Resize pred_mask_viz to match original mask size
pred_mask_viz_resized = cv2.resize(pred_mask_viz, original_size, interpolation=cv2.INTER_NEAREST)

cv2.imshow('original image', image)
cv2.waitKey(0)
cv2.imshow('original mask', mask)
cv2.waitKey(0)
cv2.imshow('predicted mask', pred_mask_viz_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()