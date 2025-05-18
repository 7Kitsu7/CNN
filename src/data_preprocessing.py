# src/data_preprocessing.py
import os
import cv2
import numpy as np
import pandas as pd

IMG_SIZE = (224, 224)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return (img / 255.0).reshape(*IMG_SIZE, 1)

def load_mask(path):
    if not os.path.exists(path):
        return np.zeros((*IMG_SIZE, 1))
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((*IMG_SIZE, 1))
    mask = cv2.resize(mask, IMG_SIZE)
    return (mask / 255.0).reshape(*IMG_SIZE, 1)

def prepare_dataset(image_dir, mask_dir, csv_path, limit=None):
    print("Preparando dataset desde:", image_dir)
    df = pd.read_csv(csv_path)
    df['filename'] = df['patientId'] + '.png'
    df = df[df['filename'].isin(os.listdir(image_dir))]

    images, masks, ages, sexes, labels = [], [], [], [], []

    count = 0
    for _, row in df.iterrows():
        if limit and count >= limit:
            break
        img_path = os.path.join(image_dir, row['filename'])
        mask_path = os.path.join(mask_dir, row['filename'])

        img = load_image(img_path)
        mask = load_mask(mask_path)

        if img is None:
            continue

        age = row['age'] if not pd.isna(row['age']) else 50
        sex = 1 if str(row['sex']).lower() == 'male' else 0
        target = int(row['Target'])

        images.append(img)
        masks.append(mask)
        ages.append(age / 100.0)
        sexes.append(sex)
        labels.append(target)

        count += 1
        if count % 1000 == 0:
            print(f"  → {count} muestras procesadas...")

    return (
        np.array(images, dtype=np.float32),
        np.array(masks, dtype=np.float32),
        np.array(list(zip(ages, sexes)), dtype=np.float32),
        np.array(labels)
    )
