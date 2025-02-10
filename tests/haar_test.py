import cv2
import numpy as np
import os
import scipy.io


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


# Load Viola-Jones
face_cascade = cv2.CascadeClassifier("../viola-johnes-classifier/cascade.xml")

# Load WIDER Face Annotations
annotations = scipy.io.loadmat("wider_face_split/wider_face_val.mat")
event_list = annotations["event_list"]
file_list = annotations["file_list"]
face_bbx_list = annotations["face_bbx_list"]

# Paths
wider_face_path = "../data/WIDER_val/images"

# Metrics counters
total_gt_faces = 0
total_detected_faces = 0
total_iou = 0
true_positives = 0
false_positives = 0
false_negatives = 0
iou_threshold = 0.5  # IoU threshold for correct detection

# Iterate over dataset
for event_idx, event in enumerate(event_list):
    event_name = event[0][0]
    image_filenames = file_list[event_idx][0]
    face_bbx = face_bbx_list[event_idx][0]

    for img_idx, image_name in enumerate(image_filenames):
        image_path = os.path.join(
            wider_face_path, event_name, image_name[0][0] + ".jpg"
        )
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ground truth faces
        gt_faces = face_bbx[img_idx][0]  # Ground truth bounding boxes

        # if (
        #     len(gt_faces) > 1
        # ):  # Skip all images that have more than one bounding box //TODO
        #     continue
        total_gt_faces += len(gt_faces)

        # Detect faces using Viola-Jones
        detected_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30)
        )
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


# Compute final statistics
precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (total_gt_faces + 1e-8)
avg_iou = total_iou / max(true_positives, 1)

# Print results
print("=== Viola-Jones Face Detection Evaluation ===")
print(f"Total Ground Truth Faces: {total_gt_faces}")
print(f"Total Detected Faces: {total_detected_faces}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Average IoU: {avg_iou:.4f}")
