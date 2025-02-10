import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.yolo import load_yolo_model, download_yolo_models
from src.constants import (
    YOLO_CFG_PATH,
    YOLO_WEIGHTS_PATH,
    YOLO_NAMES_PATH,
)
from utils import FaceDetectionAlgorithm


if __name__ == "__main__":

    # Download YOLO models if they are not present
    download_yolo_models()

    yolo_net, yolo_classes = load_yolo_model(
        YOLO_CFG_PATH, YOLO_WEIGHTS_PATH, YOLO_NAMES_PATH
    )
    output_layers = yolo_net.getUnconnectedOutLayersNames()
    conf_threshold = 0.6
    nms_threshold = 0.6

    def yolo_classifier(image):
        """
        Detects objects in an image using the YOLO model.
        Args:
            image (np.array): Input image.

        Returns:

            list: List of bounding boxes for detected objects.
        """
        height, width = image.shape[:2]

        # Preprocess the image for YOLO
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
        )
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(output_layers)

        # Initialize lists for boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Parse YOLO outputs
        for output in outputs:
            for detection in output:
                scores = detection[
                    5:
                ]  # Skip the first 5 elements (x, y, w, h, confidence)
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:  # Filter by confidence threshold
                    center_x, center_y, w, h = (
                        detection[0:4] * np.array([width, height, width, height])
                    ).astype("int")
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
            for (
                i
            ) in (
                indices.flatten()
            ):  # Flatten handles both single and multi-dimensional cases
                final_boxes.append(boxes[i])

        return final_boxes

    yolo_algorithm = FaceDetectionAlgorithm(
        classifier_name="YOLO",
        detect_faces=yolo_classifier,
    )

    result = yolo_algorithm.evaluate()
    print(result)
