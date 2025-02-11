import cv2
import numpy as np
import scipy.io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import (
    TEST_DATASET_PATH,
)


class FaceDetectionAlgorithm:
    """
    A class used for evaluating a face detection algorithm on the WIDER FACE dataset.
    It encapsulates both the classifier's name and the detection method, then calculates
    various metrics such as Precision, Recall, and Average IoU.
    """

    _classifier_name = None
    detect_faces = None

    def __init__(self, classifier_name: str, detect_faces):
        """
        Initialize the FaceDetectionAlgorithm with the given classifier name and detect_faces method.

        Args:
            classifier_name (str): Name of the face detection classifier.
            detect_faces (callable): Function or method to detect faces in an image.
        """
        self._classifier_name = classifier_name
        self.detect_faces = detect_faces

    def evaluate(self) -> str:
        """
        Test the face detection algorithm on the WIDER FACE dataset and compute statistics.

        Returns:
            str: Evaluation results including Precision, Recall, and Average IoU.
        """
        annotations = scipy.io.loadmat("wider_face_split/wider_face_val.mat")
        event_list = annotations["event_list"]
        file_list = annotations["file_list"]
        face_bbx_list = annotations["face_bbx_list"]

        total_gt_faces = 0
        total_detected_faces = 0
        total_iou = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        iou_threshold = 0.5

        for event_idx, event in enumerate(event_list):
            event_name = event[0][0]
            image_filenames = file_list[event_idx][0]
            face_bbx = face_bbx_list[event_idx][0]

            for img_idx, image_name in enumerate(image_filenames):
                image_path = os.path.join(
                    TEST_DATASET_PATH, event_name, image_name[0][0] + ".jpg"
                )
                image = cv2.imread(image_path)
                if image is None:
                    continue

                gt_faces = face_bbx[img_idx][0]
                total_gt_faces += len(gt_faces)

                detected_faces = self.detect_faces(image)
                total_detected_faces += len(detected_faces)

                tp_per_image = 0

                for detected_face in detected_faces:
                    best_iou = 0
                    for gt_face in gt_faces:
                        iou = compute_iou(detected_face, gt_face)
                        best_iou = max(best_iou, iou)

                    if best_iou >= iou_threshold:
                        tp_per_image += 1
                        total_iou += best_iou
                    else:
                        false_positives += 1

                false_negatives += len(gt_faces) - tp_per_image
                true_positives += tp_per_image

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (total_gt_faces + 1e-8)
        avg_iou = total_iou / max(true_positives, 1)

        result_str = (
            f"=== {self._classifier_name} Face Detection Evaluation ===\n"
            f"Total Ground Truth Faces: {total_gt_faces}\n"
            f"Total Detected Faces: {total_detected_faces}\n"
            f"True Positives: {true_positives}\n"
            f"False Positives: {false_positives}\n"
            f"False Negatives: {false_negatives}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"Average IoU: {avg_iou:.4f}\n"
        )
        return result_str


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
