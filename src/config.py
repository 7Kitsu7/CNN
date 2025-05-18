import os
import tensorflow as tf

class Config:
    # Configuración GPU Intel
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Limitar memoria a 4GB para evitar crashes
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            print("✅ GPU Intel configurada (4GB limit)")
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")

    # Rutas (ajustadas a tu estructura)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    
    # Parámetros del modelo
    IMG_SIZE = (128, 128)  # Tamaño reducido para tu GPU
    BATCH_SIZE = 6          # Batch size reducido
    EPOCHS = 30
    VAL_SPLIT = 0.2
    
    # Hiperparámetros
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    
    # Rutas completas
    @property
    def TRAIN_CSV(self):
        return os.path.join(self.DATA_DIR, 'stage2_train_metadata.csv')
    
    @property
    def TRAIN_IMG_DIR(self):
        return os.path.join(self.DATA_DIR, 'Training/Images')
    
    @property
    def MODEL_SAVE_DIR(self):
        path = os.path.join(self.BASE_DIR, 'model')
        os.makedirs(path, exist_ok=True)
        return path

config = Config()