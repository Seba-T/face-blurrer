import cv2

# Funzione per applicare l'effetto di blurring sui volti rilevati
def blur_faces(image, faces):
    blurred_image = image.copy()
    for (x, y, w, h) in faces:
        # Estrai la regione del volto
        face_region = blurred_image[y:y+h, x:x+w]
        
        # Applica Gaussian Blur
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
        
        # Sovrascrivi la regione offuscata sull'immagine originale
        blurred_image[y:y+h, x:x+w] = blurred_face
    return blurred_image