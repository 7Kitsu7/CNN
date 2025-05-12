# src/train_model.py
from cnn_model import build_model
from data_preprocessing import prepare_dataset
import os
import numpy as np

# Rutas
image_dir = "dataset/Training/Images"
mask_dir = "dataset/Training/Masks"
csv_path = "dataset/stage2_train_metadata.csv"

# Cargar datos
X_img, X_mask, X_tab, y = prepare_dataset(image_dir, mask_dir, csv_path)

# Construir modelo
model = build_model()

# Entrenar
model.fit([X_img, X_tab], y, batch_size=32, epochs=10, validation_split=0.2)

# Guardar
os.makedirs("model", exist_ok=True)
model.save("model/pneumonia_cnn.h5")
