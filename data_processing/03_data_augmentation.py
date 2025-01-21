import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
import albumentations as A

# Constants
IMG_H = 64
IMG_W = 64
AUG_MULTIPLIER = 3  # Numero di immagini augmentate per ogni immagine originale

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "./../data_set")

TRAIN_CSV = os.path.join(BASE_DIR, 'train_set.csv')
AUGMENTED_CSV = os.path.join(BASE_DIR, 'train_set_augmented.csv')
IMAGES_INPUT_FOLDER = os.path.join(BASE_DIR, 'images')
IMAGES_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'augmented_images')

# Crea la cartella di output se non esiste
os.makedirs(IMAGES_OUTPUT_FOLDER, exist_ok=True)

# Data augmentation pipeline
def data_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Flip orizzontale con probabilità 50%
        A.Rotate(limit=15, p=0.5),  # Rotazione tra -15° e +15°
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Rumore gaussiano
        A.RandomBrightnessContrast(p=0.5),  # Modifica casuale di luminosità e contrasto
    ])

# Carica e ridimensiona un'immagine
def load_and_resize_image(img_name):
    img_path = os.path.join(IMAGES_INPUT_FOLDER, img_name)
    img = imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {img_path}")
    # Ridimensiona l'immagine a (IMG_H, IMG_W)
    img_resized = resize(img, (IMG_H, IMG_W), anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8)  # Converti i valori in uint8

# Salva un'immagine su disco
def save_image(img, filename):
    output_path = os.path.join(IMAGES_OUTPUT_FOLDER, filename)
    imsave(output_path, img)

# Applica l'augmentation e salva i dati
def augment_and_save(csv_path):
    data = pd.read_csv(csv_path)  # Leggi il CSV
    augmented_data = []  # Lista per memorizzare i dati augmentati
    augmenter = data_augmentation_pipeline()  # Inizializza la pipeline

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing images"):
        img_name = row['filename']
        label = row['label']

        try:
            # Carica e ridimensiona l'immagine
            img_resized = load_and_resize_image(img_name)

            # Salva l'immagine originale
            original_filename = f"original_{img_name}"
            save_image(img_resized, original_filename)
            augmented_data.append([original_filename, label])

            # Genera immagini augmentate
            for i in range(AUG_MULTIPLIER):
                augmented = augmenter(image=img_resized)['image']
                augmented_filename = f"aug_{i}_{img_name}"
                save_image(augmented, augmented_filename)
                augmented_data.append([augmented_filename, label])

        except Exception as e:
            print(f"Errore durante l'elaborazione di {img_name}: {e}")

    # Salva il nuovo CSV con i dati augmentati
    augmented_df = pd.DataFrame(augmented_data, columns=["filename", "label"])
    augmented_df.to_csv(AUGMENTED_CSV, index=False)
    print(f"Augmented data saved to {AUGMENTED_CSV}")

# Esegui la funzione di augmentazione
augment_and_save(TRAIN_CSV)
