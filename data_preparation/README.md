# Data Visualization

This directory contains scripts that visualize the data, and prepare the dataset for training.

Scripts:

- visualize_coco: Loads each file from original dataset and visualize its content.
- coco_parser_bbox: Parse the original dataset and visualise the images with labels and bounding boxes.
- coco_data_visualizer: Parse the original dataset and visualise the images with labels, bounding boxes, and masks.

- preprocess_coco: Creates a new dataset for training, one for validation, each having a folder for the original images and a folder with the masks.
