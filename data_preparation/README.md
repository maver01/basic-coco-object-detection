# Data Visualization

This directory contains scripts that visualize the data, and prepare the dataset for training.

Scripts:

- visualize_coco: Loads each file from original dataset and visualize its content.
- coco_parser_bbox: Parse the original dataset and visualise the images with labels and bounding boxes.
- coco_data_visualizer: Parse the original dataset and visualise the images with labels, bounding boxes, and masks.

The preprocess_coco scripts create a new dataset for training, one for validation, each having a folder for the original images and a folder with the masks

- preprocess_coco_1: For each object in each image in the image folder, a binary mask is created.
- preprocess_coco_2: For each image in the image folder, one single mask is created. Each pixel of the mask will have a value between 0 and 90, where 0 is "not a category", the other values correspond to the categories in the dataset. Each pixel will be associated to a category (including 0 for "not a category").
