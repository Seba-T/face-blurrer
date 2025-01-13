import cv2
from blurring import blur_faces

def process_video(input_path, output_path, face_cascade_path, blur_method="gaussian"):
    """
    Processes a video to detect and blur faces frame by frame.
    
    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to the output video.
        face_cascade_path (str): Path to the Haar Cascade file.
        blur_method (str): Blurring method ('gaussian').
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
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Apply the blur method to the faces
        if blur_method == "gaussian":
            processed_frame = blur_faces(frame, faces)
        else:
            raise ValueError("Blurring effect not valid. Use 'gaussian'.")

        # Write the processed frame
        video_writer.write(processed_frame)

    # Release the VideoCapture and VideoWriter objects
    video_capture.release()
    video_writer.release()
    print(f"Processed video saved at: {output_path}")