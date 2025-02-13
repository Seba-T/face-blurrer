# Face Blurring Project

A Python project to detect faces using the Viola-Jones algorithm and apply blurring effects for privacy protection.

## Requirements

- Python 3.8+
- Opencv-python
- Numpy
- Matplotlib
- Ipykernel
- Jupyter
- Requests
- Gdown
- Scipy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To understand how to use the implemented face detection and blurring algorithms, refer to the Jupyter notebook available in the `notebook` directory.  

### Interactive Guide with Jupyter Notebook
The file `face_blur.ipynb` provides a step-by-step explanation of how to correctly use the functionalities contained in the `src` directory. It covers:  
- Loading and setting up the required models (Viola-Jones and YOLO).  
- Running face detection on images and applying different blurring techniques.  
- Performing video blurring on sample videos provided in the data/raw/ directory to test the effectiveness of the algorithms in real-world scenarios.

To launch the notebook, navigate to the project directory and run the notebook. For further details, please refer to the documentation and comments inside the notebook.

## WIDER Face Dataset for Testing

This project supports testing against the WIDER Face dataset. To enable this, you need to download the WIDER Face validation images and organize them correctly.

1. **Download:** Obtain the "WIDER Face Validation Images" from [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/).

2. **Unzip:**

   - In the root directory of your `FACE-BLURRER` project, decompress the zip file just downloaded. You should now have a directory named `WIDER_val`. If this directory already exists from a previous run and you want to use fresh data, ensure it's empty. The resulting directory structure should appear as follows:

     ```
     FACE-BLURRER/
     ├── ... other files and directories ...
     └── WIDER_val/images
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
