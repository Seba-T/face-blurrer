import cv2


# Visualize the detected faces
def draw_faces(image, faces):
    image_copy = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return image_copy


# Function to apply blurring effect on detected faces
def blur_faces(image, faces, blur_type="gaussian"):
    """
    Applies a blurring effect to the detected faces in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        faces (list): List of face bounding boxes [(x, y, w, h), ...].
        blur_type (str): The type of blur to apply. Options are "gaussian", "pixelation", or "median".

    Returns:
        numpy.ndarray: The image with blurred faces.
    """
    blurred_image = image.copy()
    for x, y, w, h in faces:
        # Extract the face region
        face_region = blurred_image[y : y + h, x : x + w]

        if blur_type == "gaussian":
            # Apply Gaussian Blur
            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
        elif blur_type == "pixelation":
            # Improved Pixelation
            height, width = face_region.shape[:2]
            # Adjust pixel size dynamically based on face size
            pixel_size = max(
                4, min(width, height) // 15
            )  # Smaller value = finer pixelation
            # Resize down to small resolution and then back up
            face_region_small = cv2.resize(
                face_region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR
            )
            blurred_face = cv2.resize(
                face_region_small, (width, height), interpolation=cv2.INTER_NEAREST
            )
        elif blur_type == "median":
            # Apply Median Blur
            blurred_face = cv2.medianBlur(face_region, 51)
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")

        # Overwrite the blurred region on the original image
        blurred_image[y : y + h, x : x + w] = blurred_face

    return blurred_image
