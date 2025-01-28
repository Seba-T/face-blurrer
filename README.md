# Face Blurring Project
A Python project to detect faces using the Viola-Jones algorithm and apply blurring effects for privacy protection.

## Requirements
- Python 3.8+
- OpenCV
- NumPy

## Installation
```bash
pip install -r requirements.txt
```

## YOLO Usage 

To use the YOLO model, you need to run the get_yolo_models.py script. This script handles downloading the required model weights and unzipping the files. Make sure all dependencies are installed.

Run the script to download and set up the YOLO model:
```bash
python get_yolo_models.py
```
Once the setup is complete, the YOLO model will be ready for use in the face detection project.