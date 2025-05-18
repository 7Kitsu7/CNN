import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.saving import register_keras_serializable
import cv2
import os
import matplotlib.pyplot as plt

# Ruta del modelo Keras preentrenado
MODEL_PATH = 'model/best_pulmonar.keras'

# Registrar funciones personalizadas
@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

@register_keras_serializable()
def jaccard_index(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

# Cargar el modelo incluyendo las funciones personalizadas
model = load_model(MODEL_PATH, custom_objects={
    'dice_coefficient': dice_coefficient,
    'jaccard_index': jaccard_index
})

def preprocess_image(image_path, target_size=(512, 512)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)# (512, 512, 1)
    img = np.expand_dims(img, axis=0)# (1, 512, 512, 1)
    return img

def segment_lungs(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)[0, :, :, 0]
    mask = (prediction > 0.5).astype(np.uint8)
    return mask

def calculate_lung_damage(mask):
    total_pixels = mask.size
    damaged_pixels = np.sum(mask)
    damage_percentage = (damaged_pixels / total_pixels) * 100
    return damage_percentage

if __name__ == '__main__':
    image_path = 'dataset/Test/neumoniasevera.png'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    
    # Segmentar la imagen
    mask = segment_lungs(image_path)
    damage_percentage = calculate_lung_damage(mask)
    # Cargar la imagen original para mostrarla junto a la máscara
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (512, 512))
    print(f"Porcentaje estimado de daño pulmonar: {damage_percentage:.2f}%")
    # Mostrar la imagen original y la máscara lado a lado
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(original_image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Máscara Segmentada")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()
