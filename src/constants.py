import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# TESTING
TESTING_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "WIDER_val/images")


# YOLO

# Directory to place all YOLO - related files
YOLO_MAIN_DIR = os.path.join(PROJECT_ROOT_DIR, "YOLO-classifier")

YOLO_CFG_PATH = os.path.join(YOLO_MAIN_DIR, "yolov3-face.cfg")
YOLO_WEIGHTS_PATH = os.path.join(YOLO_MAIN_DIR, "yolov3-wider_16000.weights")
YOLO_NAMES_PATH = os.path.join(YOLO_MAIN_DIR, "wider_face.names")


# VIOLA JONES

VIOLA_JONES_CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT_DIR, "viola-johnes-classifier/cascade.xml"
)
