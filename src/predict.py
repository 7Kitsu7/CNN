import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import cv2
import os
import matplotlib.pyplot as plt

# ========== CONFIGURACIÓN ==========
INPUT_CONFIG = {
    'image_path': 'dataset/Test/neumonia-.png',  # Ruta de la imagen
    'age': 45,                                 # Edad del paciente
    'sex': 1,                                  # 0: Femenino, 1: Masculino
    'output_dir': 'results',                   # Carpeta para guardar resultados
    'img_size': (512, 512),                    # Tamaño para segmentación pulmonar
    'clf_size': (224, 224),                    # Tamaño para clasificación
    'damage_size': (128, 128)                  # Tamaño para detección de daño
}

MODEL_PATHS = {
    'pneumonia': 'model/best_pneumonia.keras',
    'lung': 'model/best_pulmonar.keras',
    'damage': 'model/best_daño.keras'
}

# ========== FUNCIONES AUXILIARES ==========
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

def preprocess_for_model(image, target_size):
    """Preprocesamiento específico para cada modelo"""
    img = cv2.resize(image, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)  # Añade dimensión de canal

# ========== PIPELINE PRINCIPAL ==========
def main():
    # Cargar modelos
    print("⚙️ Cargando modelos...")
    try:
        models = {
            'pneumonia': load_model(MODEL_PATHS['pneumonia']),
            'lung': load_model(MODEL_PATHS['lung'], custom_objects={
                'dice_coefficient': dice_coefficient
            }),
            'damage': load_model(MODEL_PATHS['damage'])
        }
        print("✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelos: {str(e)}")
        return

    # Cargar y preprocesar imagen
    print("\n🖼️ Procesando imagen...")
    try:
        original_img = cv2.imread(INPUT_CONFIG['image_path'], cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError("No se pudo leer la imagen")
        
        # Preprocesar para cada modelo
        clf_img = preprocess_for_model(original_img, INPUT_CONFIG['clf_size'])
        seg_img = preprocess_for_model(original_img, INPUT_CONFIG['img_size'])
        
        # Preparar inputs
        clf_input = np.expand_dims(clf_img, axis=0)  # (1, 224, 224, 1)
        seg_input = np.expand_dims(seg_img, axis=0)  # (1, 512, 512, 1)
    except Exception as e:
        print(f"❌ Error procesando imagen: {str(e)}")
        return

    # Clasificación de neumonía
    print("\n🔍 Realizando diagnóstico...")
    try:
        tabular_data = np.array([[INPUT_CONFIG['age'], INPUT_CONFIG['sex']]])
        pneumonia_prob = models['pneumonia'].predict([clf_input, tabular_data])[0][0]
        diagnosis = "NEUMONÍA" if pneumonia_prob > 0.5 else "Normal"
        print(f"• Diagnóstico: {diagnosis} (Probabilidad: {pneumonia_prob:.2%})")
    except Exception as e:
        print(f"❌ Error en clasificación: {str(e)}")
        return

    # Segmentación pulmonar
    print("\n🫁 Segmentando pulmones...")
    try:
        lung_pred = models['lung'].predict(seg_input)[0, ..., 0]
        lung_mask = (lung_pred > 0.5).astype(np.uint8)
        print(f"• Área pulmonar detectada: {np.sum(lung_mask)} píxeles")
    except Exception as e:
        print(f"❌ Error en segmentación pulmonar: {str(e)}")
        return

    # Detección de daño
    print("\n⚠️ Detectando áreas dañadas...")
    try:
        # Preprocesar específicamente para el modelo 'damage' (128x128)
        damage_img = preprocess_for_model(original_img, INPUT_CONFIG['damage_size'])
        damage_input = np.expand_dims(damage_img, axis=0)  # Forma: (1, 128, 128, 1)
        
        boxes = models['damage'].predict(damage_input)[0]
        damage_mask = np.zeros(INPUT_CONFIG['img_size'], dtype=np.uint8)  # Mascara final en 512x512
        
        for box in boxes:
            if box[4] > 0.3:  # Filtro de confianza
                x, y, w, h = box[:4]
                # Escalar coordenadas a 512x512 (tamaño original para visualización)
                x1 = int(x * INPUT_CONFIG['img_size'][1])
                y1 = int(y * INPUT_CONFIG['img_size'][0])
                x2 = int((x + w) * INPUT_CONFIG['img_size'][1])
                y2 = int((y + h) * INPUT_CONFIG['img_size'][0])
                cv2.rectangle(damage_mask, (x1, y1), (x2, y2), 1, -1)
        
        print(f"• Áreas dañadas detectadas: {np.sum(damage_mask)} píxeles")
    except Exception as e:
        print(f"❌ Error en detección de daño: {str(e)}")
        return

    # Cálculo de daño
    try:
        damage_perc = (np.sum(lung_mask * damage_mask) / np.sum(lung_mask)) * 100
        print(f"\n📊 Porcentaje de daño pulmonar: {damage_perc:.2f}%")
    except:
        damage_perc = 0.0
        print("⚠️ No se pudo calcular el daño (posiblemente no se detectaron pulmones)")

    # Visualización
    print("\n🎨 Generando visualización...")
    try:
        plt.figure(figsize=(15, 5))
        
        # Imagen original
        plt.subplot(1, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Imagen Original")
        plt.axis('off')
        
        # Superposición
        plt.subplot(1, 3, 2)
        plt.imshow(original_img, cmap='gray')
        plt.imshow(np.ma.masked_where(lung_mask == 0, lung_mask), 
                  cmap='Greens', alpha=0.3)
        plt.imshow(np.ma.masked_where(damage_mask == 0, damage_mask), 
                  cmap='Reds', alpha=0.5)
        plt.title(f"Daño: {damage_perc:.2f}%")
        plt.axis('off')
        
        # Mapa de calor
        plt.subplot(1, 3, 3)
        plt.imshow(damage_mask, cmap='hot')
        plt.title(f"Prob: {pneumonia_prob:.2%}")
        plt.axis('off')
        
        # Guardar
        os.makedirs(INPUT_CONFIG['output_dir'], exist_ok=True)
        output_path = os.path.join(INPUT_CONFIG['output_dir'], 
                                 os.path.basename(INPUT_CONFIG['image_path']))
        plt.savefig(output_path, bbox_inches='tight', dpi=120)
        print(f"💾 Resultados guardados en: {output_path}")
        plt.close()
    except Exception as e:
        print(f"❌ Error generando visualización: {str(e)}")

if __name__ == '__main__':
    main()