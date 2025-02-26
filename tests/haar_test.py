import cv2
import os
import sys
from utils import FaceDetectionAlgorithm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import HAARCASCADE_CLASSIFIER_PATH


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_CLASSIFIER_PATH)

    def viola_jones_classifier(image) -> list:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces using Viola-Jones
        return face_cascade.detectMultiScale(
            gray, scaleFactor=1.07, minNeighbors=4, minSize=(30, 30)
        )

    viola_jones_algorithm = FaceDetectionAlgorithm(
        classifier_name="Custom Trained Viola-Jones",
        detect_faces=viola_jones_classifier,
    )

    result = viola_jones_algorithm.evaluate()
    print(result)
