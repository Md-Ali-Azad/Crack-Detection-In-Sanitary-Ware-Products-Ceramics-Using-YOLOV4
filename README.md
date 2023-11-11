# Crack Detection In Sanitary-Ware Products (Ceramics) Using YOLOV4
<p align="center">
    <img src="https://github.com/Md-Ali-Azad/Crack-Detection-In-Sanitary-Ware-Products-Ceramics-Using-YOLOV4/assets/42915707/8c78f65a-26a6-4089-a540-aada3f9b226f" width=350 height=310>
    <img src="https://github.com/Md-Ali-Azad/Crack-Detection-In-Sanitary-Ware-Products-Ceramics-Using-YOLOV4/assets/42915707/7be8ddc9-61a1-4fde-9e2c-d7541ef5ce85" width=350 height=310>
    <img src="https://github.com/Md-Ali-Azad/Crack-Detection-In-Sanitary-Ware-Products-Ceramics-Using-YOLOV4/assets/42915707/098a79ed-f789-4798-8efb-3b7ecdd7909f" width=350 height=310>
</p>
This project leverages the power of YOLOv4, an advanced object detection algorithm, to identify and locate cracks in ceramic sanitary-ware products in real-time. This cutting-edge solution employs deep learning techniques to enhance the quality control process. The dataset, comprising both Crack and Non-Crack images, was meticulously collected from [Euro-Bangla Ceramics Ltd.](https://eurobanglaceramicsltd.com/). I extend my sincere gratitude to the office authorities for generously granting permission to release a condensed version of the system developed exclusively for their use. The original dataset is part of an ongoing research paper that is currently being conducted. It will be made publicly available soon after the completion of the research.

# Files Description

## Overview

This repository contains a Live Crack Detection System implemented in Python using the YOLOv4 object detection model. The system is designed to detect cracks in real-time from a live video stream as well as from selected video and image files.

## Files

### 1. `Live Crack Detection.py`

This Python script enables real-time crack detection using the YOLOv4 model. It utilizes the webcam to stream live video and overlays bounding boxes around detected cracks.

### 2. `Crack Detection By Selecting Video.py`

This script allows crack detection in pre-recorded video files. Users can select a video file, and the script will process it, highlighting any detected cracks with bounding boxes.

### 3. `Crack Detection By Selecting Image.py`

This script provides crack detection functionality for static images. Users can choose an image file, and the script will identify and mark cracks in the image.

## `yolov4` Folder

The `yolov4` folder contains essential files for the YOLOv4 model, including weights, configuration (`cfg`), and class names (`names`). These files are crucial for the proper functioning of the crack detection scripts. Ensure that you have the correct versions of these files before running the scripts.

- **`weights`**: Contains the trained weights for the YOLOv4 model.
- **`cfg`**: Configuration file specifying the architecture of the YOLOv4 model.
- **`obj.names`**: Text file containing the class names used during training.

## Getting Started

To use the Live Crack Detection System:

1. Clone the repository.
2. Ensure the "yolov4" folder contains the necessary YOLOv4 files.
3. Run the desired Python script based on your use case.
4. Make sure, you are using the latest version of OpenCV.
```bash
git clone https://github.com/Md-Ali-Azad/Crack-Detection-In-Sanitary-Ware-Products-Ceramics-Using-YOLOV4.git
cd Crack-Detection-In-Sanitary-Ware-Products-Ceramics-Using-YOLOV4
