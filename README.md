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


## WIDER Face Dataset for Testing

This project supports testing against the WIDER Face dataset. To enable this, you need to download the WIDER Face validation images and organize them correctly.

1. **Download:** Obtain the "WIDER Face Validation Images" from [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/).

2. **Directory Structure:**

   - In the root directory of your `FACE-BLURRER` project, create a directory named `WIDER_val`.  If this directory already exists from a previous run and you want to use fresh data, ensure it's empty.
   - Place the downloaded validation images *directly* inside the `WIDER_val` directory.  The resulting structure should resemble the following:

     ```
     FACE-BLURRER/
     ├── ... other files and directories ...
     └── WIDER_val/
         ├── 0--Parade/
         │   ├── 0--Parade_0_904.jpg
         │   ├── 0--Parade_1_100.jpg
         │   └── ... more images ...
         ├── 1--Handclapping/
         │   ├── ... images ...
         └── ... more subdirectories ...
     ```

3. **Usage:**

   After placing the images in the correct location, you can execute the provided test scripts (e.g., `YOLO_test.py`). These scripts are designed to utilize the WIDER Face validation set. For specific execution instructions, please consult the project's documentation or the comments within the test scripts themselves.