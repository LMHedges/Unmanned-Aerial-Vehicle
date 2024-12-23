# Unmanned Aerial Vehicle (UAV) Detection

This repository contains a YOLOv8-based model for detecting both regular drones (multirotors) and fixed-wing drones in images.
<div  align="center">
<h3 align='center'>Fixed-wing UAV</h3>
<img  align="center" src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNmUzeHo3NnE3ZjM3M29tOXYwMXdiMHhpdDlhNzJmYm50MHZiaHIyNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/eWtwSaEZHRZfBU6mDR/giphy.gif">
</div>

<div align="center"><h3 align='center'>Drones (multirotors)</h3> <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmVmdHNoam1oaGY3Yjl6anY4amU5YzZpdDFiMWdtY3JvcWkzemdrayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/YFt1vuUWIUEK20032A/giphy.gif" alt="Image 1" style="display:inline-block; margin: 10px; max-width: 40% !important; height: auto;">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmttc25lZzI1NTZjemlsbXkybWNhZzdidGZnaDBxazd1bGN0M3B0diZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/F0s1DhmP8kjRyceiZn/giphy.gif" alt="Image 2" style="display:inline-block; margin: 10px; max-width: 40% !important; height: auto;">
  <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXhzaWxvZjJpcXZibG1mYW82b3RvaGczeXpnOWwxcnFzMzJjcHl2MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ZexM3mTNT6wgeme0tw/giphy.gif" alt="Image 3" style="display:inline-block; margin: 10px; max-width: 40% !important; height: auto;">
</div>


## Overview

This project aims to provide a robust solution for detecting UAVs (Unmanned Aerial Vehicles), commonly known as drones, in visual data. The model is trained using the YOLOv8 object detection framework, known for its speed and accuracy. This repository includes the trained model weights, example usage scripts, and documentation to help you get started with UAV detection.

## Requirements Installation

```bash
pip install -r requirements.txt
```
## Results and Discussion
This section presents a comprehensive evaluation of the drone detection model's performance. The model was trained five times, with each training run consisting of 10 epochs.
- Results are provided for both the training and testing datasets.

[![Alt text](https://i.postimg.cc/nhf8sW-8w/results.png)](https://github.com/Alireza0K)

## Dataset

*The training dataset contains 3,000 ***Drone*** images, evenly distributed between multirotor and ***fixed-wing*** types.*

*   **Description:** The dataset consists of images collected from various sources, including online datasets and custom captures. It contains images of both multirotor and fixed-wing drones in different environments and lighting conditions.
*   **Size:** The training set contains 1000 images, the validation set contains 200 images, and the test set contains 300 images.
*   **Classes:** The model is trained to detect two classes: 'drone' and 'fixed-wing'."

    Example:
    ```
    path: direct/project/path
    train: train
    val: valid

    names:
    0: drone
    1: Fix-wing-drone
    ```

## Model Training

The model was trained using YOLOv8 with the following parameters:

*   **Model Architecture:** YOLOv8n, YOLOv8s, YOLOv8m 
*   **Epochs:** 51
*   **Batch Size:** 16
*   **Image Size:** 640x640
*   **Device:** GPU (MPS for Apple Silicon or CUDA for NVIDIA, if used) or CPU (I used MPS)

## Training command:

```bash
python3 main.py
