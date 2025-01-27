import cv2
import numpy as np

def load_yolo_model(cfg_path, weights_path, names_path):
    """
    Load the YOLO model for face detection.
    
    Args:
        cfg_path (str): Path to the YOLO configuration file.
        weights_path (str): Path to the YOLO weights file.
        names_path (str): Path to the file with class names.
    
    Returns:
        net (cv2.dnn.Net): YOLO network.
        classes (list): List of class names.
    """
    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Load class names
    with open(names_path, "r") as f:
        classes = f.read().strip().split("\n")
    
    return net, classes


def detect_faces_yolo(image, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detects faces in an image using YOLO.
    
    Args:
        image (np.array): Input image.
        net (cv2.dnn.Net): YOLO model.
        output_layers (list): Names of the model's output layers.
        conf_threshold (float): Confidence threshold for detections.
        nms_threshold (float): Threshold for Non-Maximum Suppression.
    
    Returns:
        list: List of bounding boxes for detected faces.
    """
    height, width = image.shape[:2]
    
    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Initialize lists for boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Parse YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Skip the first 5 elements (x, y, w, h, confidence)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:  # Filter by confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Convert indices to a list of boxes
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():  # Flatten handles both single and multi-dimensional cases
            final_boxes.append(boxes[i])

    return final_boxes