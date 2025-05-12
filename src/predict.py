import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224)) / 255.0
    return img.reshape(1, 224, 224, 1)

def calculate_damage_percentage(mask_pred):
    binary_mask = (mask_pred > 0.5).astype(np.uint8)
    affected_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    return (affected_pixels / total_pixels) * 100

# === Configura aquí ===
image_path = "dataset/Test/neumonia.jpg"
age = 45
sex = 1  # 0: F, 1: M

# === Carga de modelos ===
print("Cargando modelos...")
clf_model = load_model("model/pneumonia_cnn.h5")
seg_model = load_model("model/segmentation_unet.h5")

# === Procesamiento de imagen ===
print("Procesando imagen...")
img = preprocess_image(image_path)

# === Predicción de máscara de segmentación ===
mask_pred = seg_model.predict(img)[0].reshape(224, 224)
damage_percentage = calculate_damage_percentage(mask_pred)

# === Predicción de neumonía ===
tabular_data = np.array([[age, sex]])
pneumonia_prob = clf_model.predict([img, tabular_data])[0][0]
has_pneumonia = pneumonia_prob > 0.5

# === Resultados ===
print(f"Diagnóstico: {'Neumonía' if has_pneumonia else 'No Neumonía'}")
print(f"Probabilidad de neumonía: {pneumonia_prob:.2%}")
print(f"Porcentaje estimado de daño pulmonar: {damage_percentage:.2f}%")
print("Shape de la máscara predicha:", mask_pred.shape)


