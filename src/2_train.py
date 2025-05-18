import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
import pandas as pd
import numpy as np
import cv2
from config import config

class PneumoniaDataGenerator(tf.keras.utils.Sequence):
    """Generador de datos personalizado para tu estructura"""
    def __init__(self, df, img_dir, batch_size=6, img_size=(128,128), shuffle=True):
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_df = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((len(batch_df), *self.img_size, 1), dtype=np.float32)
        y = np.zeros((len(batch_df), *self.img_size, 1), dtype=np.float32)
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            # Cargar imagen
            img_path = os.path.join(self.img_dir, f"{row['patientId']}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            X[i] = np.expand_dims(img, axis=-1) / 255.0  # Normalizar
            
            # Crear máscara si hay neumonía
            if row['Target'] == 1:
                mask = np.zeros(self.img_size, dtype=np.float32)
                x = int(row['x'] * self.img_size[1] / 1024)  # Asumiendo original 1024x1024
                y_pos = int(row['y'] * self.img_size[0] / 1024)
                w = int(row['width'] * self.img_size[1] / 1024)
                h = int(row['height'] * self.img_size[0] / 1024)
                cv2.rectangle(mask, (x,y_pos), (x+w,y_pos+h), 1.0, -1)
                y[i] = np.expand_dims(mask, axis=-1)
        
        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_optimized_unet():
    """Modelo U-Net optimizado para tu GPU Intel"""
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 1))
    
    # Encoder (Reducción de dimensionalidad)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Cuello de botella
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    
    # Decoder (Recuperación de resolución)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(x)
    
    # Salida
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return Model(inputs, outputs)

def train_model():
    print("\n=== INICIO DEL ENTRENAMIENTO ===")
    print(f"🖼️ Tamaño de imagen: {config.IMG_SIZE}")
    print(f"📦 Tamaño de lote: {config.BATCH_SIZE}")
    print(f"🔁 Épocas: {config.EPOCHS}")
    print(f"📊 División validación: {config.VAL_SPLIT*100}%")
    
    # Cargar divisiones
    train_df = pd.read_csv(os.path.join(config.MODEL_SAVE_DIR, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(config.MODEL_SAVE_DIR, 'val_split.csv'))
    
    # Crear generadores
    train_gen = PneumoniaDataGenerator(
        train_df, 
        config.TRAIN_IMG_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE
    )
    
    val_gen = PneumoniaDataGenerator(
        val_df,
        config.TRAIN_IMG_DIR,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        shuffle=False
    )
    
    # Construir modelo
    model = build_optimized_unet()
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', MeanIoU(num_classes=2)]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODEL_SAVE_DIR, 'best_daño.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Entrenamiento
    print("\n🚀 Comenzando entrenamiento...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar modelo final
    model.save(os.path.join(config.MODEL_SAVE_DIR, 'pneumonia_model.keras'))
    print(f"\n💾 Modelo guardado en: {config.MODEL_SAVE_DIR}")

if __name__ == '__main__':
    # Verificar acceso a GPU
    print(f"🖥️ Dispositivos disponibles: {tf.config.list_physical_devices()}")
    
    try:
        train_model()
        print("\n🎉 ¡Entrenamiento completado con éxito!")
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("\nPosibles soluciones:")
        print("1. Verifica que hayas ejecutado 1_preprocess.py primero")
        print("2. Confirma que las imágenes estén en la ruta correcta")
        print("3. Reduce el batch_size en config.py si hay errores de memoria")
        print("4. Verifica que tengas los archivos train_split.csv y val_split.csv")