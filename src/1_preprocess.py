import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import config

def verify_dataset():
    """Verifica acceso a datos con la nueva estructura"""
    print("\n🔍 Verificando estructura de datos...")
    
    # 1. Verificar CSV
    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(
            f"No se encontró el archivo CSV en: {config.TRAIN_CSV}\n"
            f"Por favor coloca 'stage2_train_metadata.csv' en: {config.DATA_DIR}"
        )
    
    # 2. Verificar imágenes
    if not os.path.exists(config.TRAIN_IMG_DIR):
        raise FileNotFoundError(
            f"No se encontró el directorio de imágenes en: {config.TRAIN_IMG_DIR}\n"
            f"Asegúrate que existe 'dataset/Training/Images/' con las imágenes PNG"
        )
    
    # 3. Verificar contenido del CSV
    try:
        df = pd.read_csv(config.TRAIN_CSV)
        required_cols = {'patientId', 'x', 'y', 'width', 'height', 'Target'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el CSV: {missing}")
        
        # 4. Verificar relación imágenes-CSV
        sample_id = df.iloc[0]['patientId']
        sample_img = os.path.join(config.TRAIN_IMG_DIR, f"{sample_id}.png")
        if not os.path.exists(sample_img):
            raise FileNotFoundError(
                f"Imagen de ejemplo no encontrada: {sample_img}\n"
                f"Verifica que los patientId en el CSV coincidan con los nombres de las imágenes"
            )
        
        print("✅ Estructura de datos verificada correctamente")
        return df
    
    except Exception as e:
        print("❌ Error en verificación:")
        print(f"Ruta CSV intentada: {config.TRAIN_CSV}")
        print(f"Ruta imágenes intentada: {config.TRAIN_IMG_DIR}")
        raise

def prepare_data():
    """Prepara los datos manteniendo la estructura original"""
    df = verify_dataset()
    
    # Dividir datos (80% train, 20% val)
    train_df, val_df = train_test_split(
        df,
        test_size=config.VAL_SPLIT,
        random_state=42,
        stratify=df['Target']
    )
    
    # Guardar splits para reproducibilidad
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(config.MODEL_SAVE_DIR, 'val_split.csv'), index=False)
    
    print(f"\n📊 Datos divididos:")
    print(f"- Entrenamiento: {len(train_df)} imágenes")
    print(f"- Validación: {len(val_df)} imágenes")
    print(f"💾 Divisiones guardadas en: {config.MODEL_SAVE_DIR}")
    
    return train_df, val_df

if __name__ == '__main__':
    print("\n=== Preprocesamiento de Datos ===")
    print(f"Ubicación CSV: {config.TRAIN_CSV}")
    print(f"Ubicación imágenes: {config.TRAIN_IMG_DIR}")
    
    try:
        train_df, val_df = prepare_data()
        print("\n🎉 ¡Preprocesamiento completado con éxito!")
        print("Ejecuta 2_train.py para comenzar el entrenamiento")
    except Exception as e:
        print("\n❌ Error durante el preprocesamiento:")
        print(str(e))
        print("\nPosibles soluciones:")
        print("1. Verifica que 'stage2_train_metadata.csv' esté directamente en /dataset/")
        print("2. Confirma que las imágenes estén en /dataset/Training/Images/")
        print("3. Asegúrate que los nombres en 'patientId' coincidan con los nombres de archivo de las imágenes")
        print("4. Verifica que el CSV tenga las columnas: patientId,x,y,width,height,Target")