# src/data_preprocessing.py
import os
import cv2
import numpy as np
import pandas as pd

IMG_SIZE = (224, 224)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img.reshape(*IMG_SIZE, 1)

def load_mask(path):
    if os.path.exists(path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMG_SIZE)
        return (mask / 255.0).reshape(*IMG_SIZE, 1)
    return np.zeros((*IMG_SIZE, 1))

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['sex'] = df['sex'].map({'M': 1, 'F': 0}).fillna(0.5)
    df['age'] = df['age'].fillna(df['age'].median())
    return df

def prepare_dataset(image_dir, mask_dir, csv_path, limit=1000):
    
    print("Ejecutando el script de procesamiento de datos...")

    df = pd.read_csv(csv_path)
    # Agrega la extensión .png a los patientId
    df['filename'] = df['patientId'] + '.png'
    df = df[df['filename'].isin(os.listdir(image_dir))]

    images = []
    masks = []
    ages = []
    sexes = []
    labels = []

    count = 0
    for idx, row in df.iterrows():
        if count >= limit:
            break

        img_path = os.path.join(image_dir, row['filename'])
        mask_path = os.path.join(mask_dir, row['filename'])

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, (224, 224)) / 255.0
        mask = cv2.resize(mask, (224, 224)) / 255.0

        images.append(img.reshape(224, 224, 1))
        masks.append(mask.reshape(224, 224, 1))
        ages.append(row['age'] / 100.0)
        sexes.append(1 if row['sex'].lower() == 'male' else 0)
        labels.append(row['Target'])  # Asegúrate de que 'Target' sea 0 o 1

        count += 1

    X_images = np.array(images, dtype=np.float32)
    X_masks = np.array(masks, dtype=np.float32)
    X_tabular = np.array(list(zip(ages, sexes)), dtype=np.float32)
    y = np.array(labels)

    return X_images, X_masks, X_tabular, y

