import os
import cv2
from pycocotools.coco import COCO
import ujson as json
import numpy as np
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Filter the UserWarning related to low contrast images
warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

# Specify the paths to the COCO dataset files
data_dir = "/home/maver02/Development/Datasets/COCO/"
train_dir = os.path.join(data_dir, 'train2017')
val_dir = os.path.join(data_dir, 'val2017')
annotations_dir = os.path.join(data_dir, 'annotations')
train_annotations_file = os.path.join(annotations_dir, 'instances_train2017.json')
val_annotations_file = os.path.join(annotations_dir, 'instances_val2017.json')

# Create directories for preprocessed images and masks
preprocessed_dir = os.path.join(data_dir, 'preprocess_coco_2_v1')
os.makedirs(os.path.join(preprocessed_dir, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'images'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_dir, 'val', 'masks'), exist_ok=True)

batch_size = 10  # Number of images to process before updating the progress bar

def preprocess_image(img_info, coco, data_dir, output_dir):
    """
    Processes a single image and its associated mask from the COCO dataset and saving it at destination.

    Args:
        img_info (dict): Metadata for the image, including 'file_name' and 'id'.
        coco (COCO): COCO object for accessing annotations.
        data_dir (str): Path to the directory containing the original images.
        output_dir (str): Path to the directory where processed images and masks will be saved.

    Returns:
        None
    """
    image_path = os.path.join(data_dir, img_info['file_name'])
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Load image to get dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Initialize mask with zeros
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in anns:
        category_id = ann['category_id']
        ann_mask = coco.annToMask(ann)
        mask[ann_mask > 0] = category_id  # Assign category ID to corresponding pixels
    
    # Save the preprocessed image
    cv2.imwrite(os.path.join(output_dir, 'images', img_info['file_name']), image)
    
    # Save the single mask
    mask_filename = img_info['file_name']
    cv2.imwrite(os.path.join(output_dir, 'masks', mask_filename), mask)
    

def preprocess_dataset(data_dir, annotations_file, output_dir):
    """
    Preprocesses a dataset by processing all images and their corresponding masks, and saving them at destination.

    Args:
        data_dir (str): Path to the directory containing the dataset images.
        annotations_file (str): Path to the COCO annotations file.
        output_dir (str): Path to the directory where the preprocessed dataset will be saved.

    Returns:
        None
    """
    coco = COCO(annotations_file)
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    image_infos = coco_data['images']

    total_images = len(image_infos)
    num_batches = total_images // batch_size

    # Use tqdm to create a progress bar
    progress_bar = tqdm(total=num_batches, desc='Preprocessing', unit='batch(es)', ncols=80)

    with ThreadPoolExecutor() as executor:
        for i in range(0, total_images, batch_size):
            batch_image_infos = image_infos[i:i+batch_size]
            futures = []

            for img_info in batch_image_infos:
                future = executor.submit(preprocess_image, img_info, coco, data_dir, output_dir)
                futures.append(future)

            # Wait for the processing of all images in the batch to complete
            for future in futures:
                future.result()

            progress_bar.update(1)  # Update the progress bar for each batch

    progress_bar.close()  # Close the progress bar once finished

# Preprocess the validation set (if required)
preprocess_dataset(val_dir, val_annotations_file, os.path.join(preprocessed_dir, 'val'))

# Preprocess the training set
# preprocess_dataset(train_dir, train_annotations_file, os.path.join(preprocessed_dir, 'train'))