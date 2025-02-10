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

# Function to compute Intersection over Union (IoU) between two bounding boxes
def compute_iou(boxA, boxB):
    # Compute intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute areas of both bounding boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Load YOLO model
cfg_path = "../src/model-weights/yolov3-face.cfg"  
weights_path = "../src/model-weights/yolov3-wider_16000.weights"  
names_path = "../src/model-weights/wider_face.names"  
yolo_net, yolo_classes = load_yolo_model(cfg_path, weights_path, names_path)
output_layers = yolo_net.getUnconnectedOutLayersNames()

# Load WIDER Face Annotations
annotations = scipy.io.loadmat("wider_face_split/wider_face_val.mat")
event_list = annotations["event_list"]
file_list = annotations["file_list"]
face_bbx_list = annotations["face_bbx_list"]

# Paths
wider_face_path = "../WIDER_val/images"

# Metrics counters
total_gt_faces = 0
total_detected_faces = 0
total_iou = 0
true_positives = 0
false_positives = 0
false_negatives = 0
iou_threshold = 0.5  # IoU threshold for correct detection
iou_scores = []

# Iterate over dataset
for event_idx, event in enumerate(event_list):
    event_name = event[0][0]
    image_filenames = file_list[event_idx][0]
    face_bbx = face_bbx_list[event_idx][0]

    for img_idx, image_name in enumerate(image_filenames):
        image_path = os.path.join(wider_face_path, event_name, image_name[0][0] + ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Ground truth faces
        gt_faces = face_bbx[img_idx][0]  # Ground truth bounding boxes
        total_gt_faces += len(gt_faces)

        # YOLO face detection
        detected_faces = detect_faces_yolo(image, yolo_net, output_layers)
        total_detected_faces += len(detected_faces)

        tp_per_image = 0  # Track true positives per image

        for detected_face in detected_faces:
            best_iou = 0
            for gt_face in gt_faces:
                iou = compute_iou(detected_face, gt_face)
                best_iou = max(best_iou, iou)

            if best_iou >= iou_threshold:
                tp_per_image += 1  # Only count for this image
                total_iou += best_iou
            else:
                false_positives += 1

        false_negatives += len(gt_faces) - tp_per_image  # Use per-image TP count
        true_positives += tp_per_image

# Compute final evaluation metrics
precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (total_gt_faces + 1e-8)
avg_iou = total_iou / max(true_positives, 1)

# Print results
print("\n=== YOLO Face Detection Evaluation ===")
print(f"Total Ground Truth Faces: {total_gt_faces}")
print(f"Total Detected Faces: {total_detected_faces}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Average IoU: {avg_iou:.4f}")


