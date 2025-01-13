import cv2

# Visualizza l'immagine con i volti rilevati
def draw_faces(image, faces):
    image_copy = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Rettangolo blu
    return image_copy