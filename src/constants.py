import os
import cv2

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# TEST DATASET
TEST_DATASET_PATH = os.path.join(PROJECT_ROOT_PATH, "WIDER_val/images")

#  RAW DATA PATH
RAW_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data/raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data/processed")

# YOLO

# Directory to place all YOLO - related files
YOLO_MAIN_DIR = os.path.join(PROJECT_ROOT_PATH, "YOLO-classifier")

YOLO_CFG_PATH = os.path.join(YOLO_MAIN_DIR, "yolov3-face.cfg")
YOLO_WEIGHTS_PATH = os.path.join(YOLO_MAIN_DIR, "yolov3-wider_16000.weights")
YOLO_NAMES_PATH = os.path.join(YOLO_MAIN_DIR, "wider_face.names")


# HAAR CASCADE (VIOLA JONES) CLASSIFIER

HAARCASCADE_CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT_PATH, "haar-cascade-classifier/cascade.xml"
)

PRE_TRAINED_HAARCASCADE_CLASSIFIER_PATH = (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
