# Unmanned Aerial Vehicle (UAV) Detection

This repository contains a YOLOv8-based model for detecting both regular drones (multirotors) and fixed-wing drones in images.

## Overview

This project aims to provide a robust solution for detecting UAVs (Unmanned Aerial Vehicles), commonly known as drones, in visual data. The model is trained using the YOLOv8 object detection framework, known for its speed and accuracy. This repository includes the trained model weights, example usage scripts, and documentation to help you get started with UAV detection.

## Dataset

*(Add details about your dataset here. This is crucial for reproducibility and understanding the model's limitations.)*

*   **Description:** Briefly describe the dataset used for training. For example: "The dataset consists of images collected from various sources, including online datasets and custom captures. It contains images of both multirotor and fixed-wing drones in different environments and lighting conditions."
*   **Size:** Specify the number of images used for training, validation, and testing (if applicable). For example: "The training set contains 1000 images, the validation set contains 200 images, and the test set contains 300 images."
*   **Classes:** List the classes the model was trained to detect. For example: "The model is trained to detect two classes: 'multirotor' and 'fixed-wing'."
*   **Annotation Format:** Specify the annotation format used (e.g., YOLO format).

    Example:
    ```
    train: path/to/train/images
    val: path/to/valid/images
    nc: 2
    names: ['multirotor', 'fixed-wing']
    ```

## Model Training

The model was trained using YOLOv8 with the following parameters:

*   **Model Architecture:** YOLOv8n, YOLOv8s, YOLOv8m (Specify which one you used or if it's a custom architecture)
*   **Epochs:** 51
*   **Batch Size:** 16
*   **Image Size:** 640x640
*   **Device:** GPU (MPS for Apple Silicon or CUDA for NVIDIA, if used) or CPU (Specify which was used)
*   **Optimizer:** (Specify the optimizer used, e.g., Adam, SGD)
*   **Loss Function:** (Specify the loss function used, e.g., CIoU, BBox Loss, Class Loss)
*   **Other Hyperparameters:** (Include any other relevant hyperparameters you used during training.)

Example training command:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=51 imgsz=640 batch=16 device=mps # or device=cuda or device=cpu