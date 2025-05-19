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
    
    def predict_single_image(self, image_path):
        """Realizar predicción en una sola imagen"""
        print(f"\n=== PREDICCIÓN PARA IMAGEN: {image_path} ===")
        
        # Verificar si la imagen existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
        
        # Obtener el nombre del archivo sin extensión
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Buscar en el dataframe si la imagen tiene anotaciones
        sample = None
        if image_name in self.val_df['patientId'].values:
            sample = self.val_df[self.val_df['patientId'] == image_name].iloc[0]
        
        # Cargar y preprocesar imagen
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
        original_size = img.shape[:2]  # Guardar tamaño original
        img_resized = cv2.resize(img, config.IMG_SIZE)
        img_input = np.expand_dims(img_resized, axis=(0, -1)) / 255.0
        
        # Predecir
        pred = self.model.predict(img_input)[0]
        pred_mask = (pred > 0.1).astype(np.uint8)
        
        # Redimensionar la máscara de predicción al tamaño original
        pred_mask_original = cv2.resize(pred_mask.squeeze(), (original_size[1], original_size[0]))
        
        # Crear máscara real (ground truth) si existe en los datos
        true_mask = None
        if sample is not None:
            true_mask = np.zeros(config.IMG_SIZE, dtype=np.uint8)
            if sample['Target'] == 1:
                x = int(sample['x'] * config.IMG_SIZE[1] / 1024)
                y = int(sample['y'] * config.IMG_SIZE[0] / 1024)
                w = int(sample['width'] * config.IMG_SIZE[1] / 1024)
                h = int(sample['height'] * config.IMG_SIZE[0] / 1024)
                cv2.rectangle(true_mask, (x,y), (x+w,y+h), 1, -1)
        
        # Visualización
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Imagen Original\n{os.path.basename(image_path)}")
        plt.axis('off')
        
        if true_mask is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title("Máscara Real (si existe)")
            plt.axis('off')
        else:
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, "No hay máscara real\ndisponible", 
                    ha='center', va='center')
            plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title("Predicción del Modelo")
        plt.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(config.MODEL_SAVE_DIR, f'prediction_{image_name}.png')
        plt.savefig(output_path)
        print(f"\n💾 Visualización guardada en: {output_path}")
        plt.show()
        
        return {
            'image': img,
            'prediction': pred_mask_original,
            'ground_truth': true_mask,
            'prediction_resized': pred_mask
        }

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
        
        # 1. Evaluación cuantitativa (opcional)
        # evaluator.evaluate_model()
        
        # 2. Definir la ruta de la imagen directamente en el código
        image_path = "dataset/Test/neumoniasevera.png"  # <-- MODIFICA AQUÍ CON TU RUTA
        
        # 3. Realizar predicción en la imagen especificada
        evaluator.predict_single_image(image_path)
        
        print("\n🎉 Evaluación completada con éxito!")
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {str(e)}")
        print("\nPosibles soluciones:")
        print("1. Verifica que la ruta de la imagen sea correcta")
        print("2. Confirma que existe best_model.keras en la carpeta models/")
        print("3. Asegúrate que los archivos de validación estén en su lugar")
        print("4. Revisa que las rutas en config.py sean correctas")