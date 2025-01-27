import gdown
import zipfile
import os

# ID del file su Google Drive per i due file
file_ids = {
    "yolov3": "13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX",  # yolov3-wider_16000.weights.zip
    "yoloface": "1a_pbXPYNj7_Gi6OxUqNo_T23Dt_9CzOV"  # YOLO_Face.h5.zip
}

output_files = {
    "yolov3": "yolov3-wider_16000.weights.zip",
    "yoloface": "YOLO_Face.h5.zip"
}

output_dir = "./src/model-weights"

# Funzione per scaricare il file da Google Drive usando gdown
def download_file(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading file from {url}...")
    gdown.download(url, output_file, quiet=False)
    print(f"Downloaded {output_file}")

# Funzione per estrarre il file ZIP
def unzip_file(zip_file, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    print(f"Extracting {zip_file} into {extract_dir}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extraction complete.")

# Scarica e decomprimi i file per ogni ID
for key in file_ids:
    # Scarica il file
    download_file(file_ids[key], output_files[key])
    
    # Estrai il file ZIP
    unzip_file(output_files[key], output_dir)
    
    # Rimuovi il file ZIP
    os.remove(output_files[key])
    print(f"{output_files[key]} removed.")

print("All files downloaded and extracted successfully!")
