import cv2
from utils import FaceDetectionAlgorithm
from src.constants import PRE_TRAINED_HAARCASCADE_CLASSIFIER_PATH

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(PRE_TRAINED_HAARCASCADE_CLASSIFIER_PATH)

    def preTrained_viola_jones_classifier(image) -> list:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces using Viola-Jones
        return face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30)
        )

    viola_jones_algorithm = FaceDetectionAlgorithm(
        classifier_name="Viola-Jones",
        detect_faces=preTrained_viola_jones_classifier,
    )

    result = viola_jones_algorithm.evaluate()
