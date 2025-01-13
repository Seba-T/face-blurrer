import cv2

# Funzione per rilevare volti
def detect_faces(image):
    # Path al file Haar Cascade
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Crea il rilevatore di volti
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Converti l'immagine in scala di grigi (richiesto dall'algoritmo Haar)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti
    faces = face_cascade.detectMultiScale(
        gray_image,       # Immagine in scala di grigi
        scaleFactor=1.1,  # Fattore di riduzione della scala
        minNeighbors=5,   # Numero minimo di vicini per rettangolo di rilevamento
        minSize=(30, 30)  # Dimensione minima del volto
    )
    return faces

