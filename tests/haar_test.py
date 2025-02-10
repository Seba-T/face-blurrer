import cv2
import os
import sys
from utils import FaceDetectionAlgorithm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import VIOLA_JONES_CLASSIFIER_PATH

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
