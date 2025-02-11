import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import blur_faces, BlurMethod
from src.constants import PRE_TRAINED_HAARCASCADE_CLASSIFIER_PATH


# Function to detect faces
def detect_faces(image, face_cascade_path=PRE_TRAINED_HAARCASCADE_CLASSIFIER_PATH):
    """
    Detect faces in an image using the given Haar Cascade classifier.

    Parameters:
        image (numpy.ndarray): The input image.
        face_cascade_path (str): Path to the Haar Cascade file.

    Returns:
        list of tuple: A list of bounding boxes for detected faces.
    """

    # Create the face detector
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Convert the image to grayscale (required by the Haar algorithm)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_image,  # Grayscale image
        scaleFactor=1.1,  # Scale reduction factor
        minNeighbors=5,  # Minimum number of neighbors per detection rectangle
        minSize=(30, 30),  # Minimum face size
    )
    return faces


# Function to process a video
def process_video_haar_cascade(
    input_path, output_path, face_cascade_path, blur_method=BlurMethod.GAUSSIAN
):
    """
    Processes a video to detect and blur faces frame by frame.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to the output video.
        face_cascade_path (str): Path to the Haar Cascade file.
        blur_method (BlurMethod): Blurring method ('BlurMethod.GAUSSIAN').
    """
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Open the video file
    video_capture = cv2.VideoCapture(input_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Video Codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # No more frames to read

        # Find faces in frames
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Apply the blur method to the faces
        processed_frame = blur_faces(frame, faces, blur_method)

        # Write the processed frame
        video_writer.write(processed_frame)

    # Release the VideoCapture and VideoWriter objects
    video_capture.release()
    video_writer.release()
    print(f"Processed video saved at: {output_path}")
