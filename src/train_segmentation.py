import os
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Paths
IMAGE_DIR = "dataset/Training/Images"
MASK_DIR = "dataset/Training/Masks"

IMG_HEIGHT, IMG_WIDTH = 224, 224  # Tamaño estándar

# Función para cargar imágenes y máscaras
def load_data(image_dir, mask_dir, limit=None):
    images, masks = [], []
    files = os.listdir(image_dir)[:limit] if limit else os.listdir(image_dir)
    for filename in files:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), np.array(masks)

# Arquitectura U-Net
def build_unet(input_shape=(224, 224, 1)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = UpSampling2D()(c3)
    u1 = Concatenate()([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = UpSampling2D()(c4)
    u2 = Concatenate()([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cargar datos
print("Cargando datos de imágenes y máscaras...")
X, Y = load_data(IMAGE_DIR, MASK_DIR, limit=1000)  # Solo 1000 para prueba rápida
print("Total de muestras cargadas:", X.shape[0])

# Separar en entrenamiento y validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = build_unet()
print("Entrenando modelo U-Net...")
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=8)

# Guardar el modelo
model.save("model/segmentation_unet.h5")
print("Modelo de segmentación guardado como 'segmentation_unet.h5'")
