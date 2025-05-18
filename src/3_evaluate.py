import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import config

class PneumoniaEvaluator:
    def __init__(self):
        # Cargar modelo y datos
        self.model = load_model(os.path.join(config.MODEL_SAVE_DIR, 'best_daño.keras'))
        self.val_df = pd.read_csv(os.path.join(config.MODEL_SAVE_DIR, 'val_split.csv'))
        
    def evaluate_model(self):
        """Evaluación cuantitativa del modelo"""
        print("\n=== EVALUACIÓN CUANTITATIVA ===")
        
        # Crear generador de validación
        val_gen = PneumoniaDataGenerator(
            self.val_df,
            config.TRAIN_IMG_DIR,
            batch_size=1,  # Usamos batch=1 para evaluación precisa
            img_size=config.IMG_SIZE,
            shuffle=False
        )
        
        # Evaluar métricas
        results = self.model.evaluate(val_gen, verbose=1)
        
        print("\n📊 Métricas de evaluación:")
        print(f"- Pérdida: {results[0]:.4f}")
        print(f"- Exactitud: {results[1]:.4f}")
        print(f"- IoU: {results[2]:.4f}")
        
        return results
    
    def visualize_predictions(self, num_samples=3):
        """Visualización de predicciones"""
        print("\n=== VISUALIZACIÓN DE PREDICCIONES ===")
        
        # Configurar matplotlib
        plt.figure(figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # Seleccionar muestra aleatoria
            sample = self.val_df.sample(1).iloc[0]
            img_path = os.path.join(config.TRAIN_IMG_DIR, f"{sample['patientId']}.png")
            
            # Cargar y preprocesar imagen
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, config.IMG_SIZE)
            img_input = np.expand_dims(img, axis=(0, -1)) / 255.0
            
            # Predecir
            pred = self.model.predict(img_input)[0]
            pred_mask = (pred > 0.5).astype(np.uint8)
            
            # Crear máscara real (ground truth)
            true_mask = np.zeros(config.IMG_SIZE, dtype=np.uint8)
            if sample['Target'] == 1:
                x = int(sample['x'] * config.IMG_SIZE[1] / 1024)
                y = int(sample['y'] * config.IMG_SIZE[0] / 1024)
                w = int(sample['width'] * config.IMG_SIZE[1] / 1024)
                h = int(sample['height'] * config.IMG_SIZE[0] / 1024)
                cv2.rectangle(true_mask, (x,y), (x+w,y+h), 1, -1)
            
            # Visualización
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Imagen {sample['patientId']}\nTarget: {sample['Target']}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(true_mask, cmap='gray')
            plt.title("Máscara Real")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(pred_mask.squeeze(), cmap='gray')
            plt.title("Predicción del Modelo")
            plt.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(config.MODEL_SAVE_DIR, 'predictions.png')
        plt.savefig(output_path)
        print(f"\n💾 Visualizaciones guardadas en: {output_path}")
        plt.show()

class PneumoniaDataGenerator(tf.keras.utils.Sequence):
    """Generador de datos para evaluación (similar al de entrenamiento)"""
    def __init__(self, df, img_dir, batch_size=1, img_size=(128,128), shuffle=False):
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
            img_path = os.path.join(self.img_dir, f"{row['patientId']}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            X[i] = np.expand_dims(img, axis=-1) / 255.0
            
            if row['Target'] == 1:
                mask = np.zeros(self.img_size, dtype=np.float32)
                x = int(row['x'] * self.img_size[1] / 1024)
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

if __name__ == '__main__':
    print("\n=== EVALUACIÓN DEL MODELO DE NEUMONÍA ===")
    print(f"📂 Ruta del modelo: {config.MODEL_SAVE_DIR}")
    
    try:
        evaluator = PneumoniaEvaluator()
        
        # 1. Evaluación cuantitativa
        evaluator.evaluate_model()
        
        # 2. Visualización cualitativa
        evaluator.visualize_predictions(num_samples=3)
        
        print("\n🎉 Evaluación completada con éxito!")
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {str(e)}")
        print("\nPosibles soluciones:")
        print("1. Verifica que hayas ejecutado primero 1_preprocess.py y 2_train.py")
        print("2. Confirma que existe best_model.keras en la carpeta models/")
        print("3. Asegúrate que los archivos de validación estén en su lugar")
        print("4. Revisa que las rutas en config.py sean correctas")