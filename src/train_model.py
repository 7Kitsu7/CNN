# src/train_model.py

from cnn_model import build_model
from data_preprocessing import prepare_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.saving import save_model  # Keras moderno
import os

# Rutas
image_dir = "dataset/Training/Images"
mask_dir = "dataset/Training/Masks"
csv_path = "dataset/stage2_train_metadata.csv"

# Cargar datos
X_img, X_mask, X_tab, y = prepare_dataset(image_dir, mask_dir, csv_path, limit=8000)

# Dividir en entrenamiento y validación
Ximg_train, Ximg_val, Xtab_train, Xtab_val, y_train, y_val = train_test_split(
    X_img, X_tab, y, test_size=0.2, random_state=42
)

# Construir modelo
model = build_model()

# Asegurar carpeta de salida
os.makedirs("model", exist_ok=True)

# Callback para guardar el mejor modelo (opcional pero recomendado)
callbacks = [
    ModelCheckpoint("model/best_pneumonia_model.keras", save_best_only=True)
]

# Entrenamiento: SIN EarlyStopping
model.fit(
    [Ximg_train, Xtab_train], y_train,
    validation_data=([Ximg_val, Xtab_val], y_val),
    batch_size=32,
    epochs=30,
    callbacks=callbacks,
    shuffle=True
)

# Guardar modelo final entrenado (aunque no sea el mejor)
save_model(model, "model/pneumonia_cnn.keras")
print("✅ Entrenamiento completo. Modelo guardado en formato moderno Keras.")
